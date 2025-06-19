from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import re
import os
import statistics
import transformers
import random
import pandas as pd
import math
import wandb
from models import DualEncoderModel, ModelLoader

def prepare_test_batch(raw_batch, negative_samples):
    queries = []
    good_documents = []
    bad_documents = []
    document_count_for_each_query = []

    for query_row in raw_batch:
        queries.append(query_row["tokenized_query"])
        good_passages = query_row["tokenized_passages"]
        good_documents.extend(good_passages)
        
        good_passage_count = len(good_passages)
        document_count_for_each_query.append(good_passage_count)

    # TODO: Add in retrying of query-id clashes
    negative_sample_choices = torch.randint(low=0, high=len(negative_samples), size=(len(good_documents),)).tolist()
    bad_documents = [
        negative_samples[i]["tokenized_passage"] for i in negative_sample_choices
    ]
    
    return {
        "tokenized_queries": queries,
        "tokenized_good_documents": good_documents,
        "tokenized_bad_documents": bad_documents,
        "document_count_for_each_query": document_count_for_each_query,
    }

def calculate_triplet_loss(query_vectors, good_document_vectors, bad_document_vectors, margin: float = 0.2):
    """
    Calculate the loss for a single query-document pair. 
    """
    good_similarity = F.cosine_similarity(query_vectors, good_document_vectors, dim=1)
    bad_similarity = F.cosine_similarity(query_vectors, bad_document_vectors, dim=1)

    diff = good_similarity - bad_similarity
    return torch.max(torch.tensor(0), torch.tensor(margin) - diff).sum(dim=0)


def process_test_batch(batch, negative_samples, model: DualEncoderModel, margin):
    prepared = prepare_test_batch(batch, negative_samples)

    query_vectors = model.embed_tokenized_queries(prepared["tokenized_queries"])                  # tensor of shape (Q, E)
    good_document_vectors = model.embed_tokenized_documents(prepared["tokenized_good_documents"]) # tensor of shape (D, E)
    bad_document_vectors = model.embed_tokenized_documents(prepared["tokenized_bad_documents"])   # tensor of shape (D, E)

    # Duplicate each query so that it pairs with each document under that query
    # This means it ends up with the same length as the good/bad document vectors
    queries_for_each_document = torch.concat([ # tensor of shape (D, E) where D is the total number of documents
        query_vector.repeat(document_count, 1) # tensor of shape (K, E) where K is the number of documents for this query
        for (query_vector, document_count) in zip(query_vectors, prepared["document_count_for_each_query"])
    ])

    total_loss = calculate_triplet_loss(
        queries_for_each_document,
        good_document_vectors,
        bad_document_vectors,
        margin
    )

    return {
        "total_loss": total_loss,
        "document_count": good_document_vectors.shape[0],
    }

class ModelTrainer:
    def __init__(
            self,
            model: DualEncoderModel,
            start_epoch = None,
            start_optimizer_state = None,
            override_to_epoch: int = None,
            validate_and_save_after_epochs: int = 5,
        ):
        torch.manual_seed(42)

        datasets.config.IN_MEMORY_MAX_SIZE = 8 * 1024 * 1024 # 8GB
        dataset = datasets.load_dataset("microsoft/ms_marco", "v1.1")

        self.model = model
        self.validate_and_save_after_epochs = validate_and_save_after_epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)

        if start_epoch is not None:
            assert start_optimizer_state is not None, "If start epoch is provided, optimizer state must also be provided"
            self.epoch = start_epoch
            self.optimizer.load_state_dict(start_optimizer_state)
            print(f"Resuming training from epoch {self.epoch}")

        if override_to_epoch is not None:
            self.model.training_parameters.epochs = override_to_epoch
            print(f"Overriding training end epoch to {self.model.training_parameters.epochs}")

        print("Pre-tokenizing queries and documents...")
        train_dataset = dataset["train"].map(
            lambda x: {
                "tokenized_query": model.tokenize_query(x["query"]),
                "tokenized_passages": [
                    model.tokenize_document(passage) for passage in x["passages"]["passage_text"]
                ],
            })

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=model.training_parameters.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Specify identity collate function (no magic batching which breaks)
        )

        self.validation_dataset = dataset["validation"].map(
            lambda x: {
                "tokenized_query": model.tokenize_query(x["query"]),
                "tokenized_passages": [
                    model.tokenize_document(passage) for passage in x["passages"]["passage_text"]
                ],
            })
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=model.training_parameters.batch_size,
            collate_fn=lambda x: x,  # Specify identity collate function (no magic batching which breaks)
        )

        print(f"Generating negative samples...")
        self.negative_samples = [
            { "tokenized_passage": tokenized_passage, "query_id": query_row["query_id"] }
            for query_row in train_dataset
            for tokenized_passage in query_row["tokenized_passages"]
        ]

    def train(self):
        print("Beginning training...")

        self.epoch = 1
        last_epoch_results = None

        while self.epoch <= self.model.training_parameters.epochs:
            print(f"Epoch {self.epoch}/{self.model.training_parameters.epochs}")
            last_epoch_results = self.train_epoch()
            if self.epoch % self.validate_and_save_after_epochs == 0 or self.epoch == self.model.training_parameters.epochs:
                self.model.validation_metrics = self.validate()

                if wandb.run is not None:
                    wandb.log({
                        "epoch": self.epoch,
                        "validation_any_relevant_result": self.model.validation_metrics["any_relevant_result"],
                        "validation_reciprical_rank": self.model.validation_metrics["reciprical_rank"],
                        "validation_average_relevance": self.model.validation_metrics["average_relevance"],
                    })
            self.save_model()
            self.epoch += 1

        print("Training complete.")
        return {
            "total_epochs": self.model.training_parameters.epochs,
            "validation": self.model.validation_metrics,
            "last_epoch": last_epoch_results,
        }

    def train_epoch(self):
        self.model.train()

        print_every = 10
        running_loss = 0.0
        running_samples = 0
        total_batches = len(self.train_data_loader)

        epoch_loss = 0.0
        epoch_samples = 0
        for batch_idx, raw_batch in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            batch_results = process_test_batch(raw_batch, self.negative_samples, self.model, self.model.training_parameters.margin)
            loss = batch_results["total_loss"]
            running_samples += batch_results["document_count"]
            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_samples += batch_results["document_count"]

            loss.backward()
            self.optimizer.step()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0:
                print(f"Epoch {self.epoch}, Batch {batch_num}/{total_batches}, Average Loss: {(running_loss / running_samples):.4f}")
                running_loss = 0.0
                running_samples = 0

        average_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        print(f"Epoch {self.epoch} complete (Average Loss: {average_loss:.4f})")

        return {
            "average_loss": average_loss,
        }

    def validate(self):
        print()
        print("== VALIDATING MODEL ==")
        print()
        document_passage_to_document_index = {} # passage => index
        document_index_to_document_ids = []
        document_index_to_document_text = []
        document_index_to_tokenized_doc = []
        total_documents = 0
    
        query_passage_to_query_index = {} # passage => query_index
        query_index_to_query_ids = []
        query_index_to_query_text = []
        query_index_to_tokenized_query = []

        print("Preparing validation data...")
        for query_row in self.validation_dataset:
            query_id = query_row['query_id']

            # Add passages
            for i, (tokenized_document, document_text) in enumerate(zip(query_row["tokenized_passages"], query_row["passages"]["passage_text"])):
                total_documents += 1
                if document_text in document_passage_to_document_index:
                    document_index = document_passage_to_document_index[document_text]
                else:
                    document_index = len(document_passage_to_document_index)
                    document_passage_to_document_index[document_text] = document_index
                    document_index_to_document_ids.append([])
                    document_index_to_document_text.append(document_text)
                    document_index_to_tokenized_doc.append(tokenized_document)
    
                document_index_to_document_ids[document_index].append((query_id, i))

            query_text = query_row['query']
            tokenized_query = query_row['tokenized_query']
            if query_text in query_passage_to_query_index:
                query_index = query_passage_to_query_index[query_text]
            else:
                query_index = len(query_passage_to_query_index)
                query_passage_to_query_index[query_text] = query_index
                query_index_to_query_ids.append([])
                query_index_to_query_text.append(query_text)
                query_index_to_tokenized_query.append(tokenized_query)

            query_index_to_query_ids[query_index].append(query_id)

        distinct_query_count = len(query_index_to_tokenized_query)
        distinct_document_count = len(document_index_to_tokenized_doc)
        print(f"Distinct queries: {distinct_query_count} (of {len(self.validation_dataset)} total queries)")
        print(f"Distinct documents: {distinct_document_count} (of {total_documents} total documents)")

        print("Generating embeddings for validation data...")
        self.model.eval()
        
        query_embeddings = self.model.embed_tokenized_queries(query_index_to_tokenized_query)
        document_embeddings = self.model.embed_tokenized_documents(document_index_to_tokenized_doc)

        k_samples = 5
        total_queries_to_consider = 1000
        print(f"Generating top-{k_samples} docs for {total_queries_to_consider} queries in validation set...")

        average_relevance_by_query = []
        reciprical_ranks_of_first_relevant_result_by_query = []
        any_relevant_result_by_query = []

        query_to_print = random.randint(0, total_queries_to_consider - 1)

        for query_index, query_embedding in enumerate(query_embeddings[:total_queries_to_consider]):
            similarities = F.cosine_similarity(query_embedding, document_embeddings, dim=1).tolist() # (D) [5, 7, 2]
            top_k_most_similar = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:k_samples]

            query_ids = query_index_to_query_ids[query_index]
            query_ids_set = set(query_ids)
            top_k_is_relevant = [
                any(document_id[0] in query_ids_set for document_id in document_index_to_document_ids[document_index])
                for document_index, _ in top_k_most_similar
            ]

            inverse_first_relevant_rank = 0
            for i in range(k_samples):
                if top_k_is_relevant[i]:
                    inverse_first_relevant_rank = 1 / (i + 1)
                    break
            reciprical_ranks_of_first_relevant_result_by_query.append(inverse_first_relevant_rank)
            average_relevance_by_query.append(statistics.mean(top_k_is_relevant))
            any_relevant_result_by_query.append(any(top_k_is_relevant))

            if query_index == query_to_print:
                print()
                query_text = query_index_to_query_text[query_index]
                print(f"Example query {query_ids} \"{query_text}\" top {k_samples} most similar documents:")
                for document_index, score in top_k_most_similar:
                    document_ids = document_index_to_document_ids[document_index]
                    passage = document_index_to_document_text[document_index]
                    print(f"  => {document_ids} with score {score:.3f}: \"{passage}\"")
                print()

        any_relevant_result = statistics.mean(any_relevant_result_by_query)
        reciprical_rank = statistics.mean(reciprical_ranks_of_first_relevant_result_by_query)
        average_relevance = statistics.mean(average_relevance_by_query)
        
        print(f"Across the first {total_queries_to_consider} queries, we queried the top {k_samples} documents (from {distinct_document_count} docs across {distinct_query_count} queries):")
        print(f"> Proportion with any relevant result     : {any_relevant_result:.2%}")
        print(f"> Reciprical rank of first relevant result: {reciprical_rank:.2%}")
        print(f"> Average relevance of results            : {average_relevance:.2%}")
        print()
        print(f"Validation complete")
        print()

        return {
            "any_relevant_result": any_relevant_result,
            "reciprical_rank": reciprical_rank,
            "average_relevance": average_relevance,
        }
    
    def save_model(self):
        model_loader = ModelLoader()
        model_loader.save_model_data(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
        )
