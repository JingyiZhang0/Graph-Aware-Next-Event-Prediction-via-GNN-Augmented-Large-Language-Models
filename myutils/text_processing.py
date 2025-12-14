# Title: text_processing.py
# Author: Jingyi Zhang
# Date: 2025-12-12
# Filename: myutils/text_processing.py
# Description: Text cleaning and embedding utilities for activity sequences.

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file, load_file
import json
from typing import Dict, List, Union, Optional
from datetime import datetime
import os
import re

class TextProcessor:
    def __init__(
        self,
        source: str = 'default',
        model_name: str = 'prajjwal1/bert-medium',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        load_model: bool = True,
    ) -> None:
        """
        Initialize the text processor.

        Args:
            source (str): Dataset source key affecting cleaning rules.
            model_name (str): HF or SentenceTransformer model name.
            device (str): 'cuda' or 'cpu'.
            load_model (bool): Whether to load the model immediately.
        """
        self.source = source.lower()
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embed_dim = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        if 'MiniLM' in self.model_name:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embed_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.embed_dim = self.model.config.hidden_size

    def clean_activity_names(self, raw_names: List[str]) -> Dict[str, str]:
        """
        Clean activity names and generate natural language descriptions.
        Applies both generic and dataset-specific rules.

        Args:
            raw_names (List[str]): Raw activity names.

        Returns:
            Dict[str, str]: Mapping from raw activity name to cleaned description.
        """
        act_to_text = {}
        for name in raw_names:
            name_proc = self._clean_generic(name)
            name_final = self._apply_dataset_specific(name_proc)
            act_to_text[name] = name_final
        return act_to_text

    def _clean_generic(self, name: str) -> str:
        """
        Generic cleaning logic applied to all datasets.

        Args:
            name (str): Raw activity name.

        Returns:
            str: Cleaned activity text.
        """
        # Remove "O_" prefix if present
        if name.startswith("O_"):
            name = name[2:]
        # Replace underscores and lower-case
        cleaned = name.replace('_', ' ').replace('-', ' ').strip().lower()
        # Replace common abbreviations
        cleaned = re.sub(r'\bsw\b', 'software', cleaned)
        return cleaned

    def _apply_dataset_specific(self, cleaned: str) -> str:
        """
        Apply domain-specific logic based on dataset source.

        Args:
            cleaned (str): Cleaned activity text.

        Returns:
            str: Dataset-specific rewritten text.
        """
        if self.source.startswith('helpdesk'):
            cleaned = cleaned.replace('assign seriousness', 'Assign seriousness level')
            cleaned = cleaned.replace('take in charge ticket', 'Take in charge of the ticket')
            cleaned = cleaned.replace('resolve ticket', 'Resolve the ticket')
            cleaned = cleaned.replace('closed', 'Close the ticket')
            cleaned = cleaned.replace('require upgrade', 'Require system upgrade')
            cleaned = cleaned.replace('wait', 'Wait for further actions')
            cleaned = cleaned.replace('insert ticket', 'Insert a new ticket')
            cleaned = cleaned.replace('create sw anomaly', 'Create software anomaly report')
            cleaned = cleaned.replace('schedule intervention', 'Schedule intervention')
            cleaned = cleaned.replace('resolved', 'Mark as resolved')
            cleaned = cleaned.replace('invalid', 'Mark as invalid')
            cleaned = cleaned.replace('verified', 'Verify the solution')
            cleaned = cleaned.replace('resolve sw anomaly', 'Resolve software anomaly')
            cleaned = cleaned.replace('duplicate', 'Mark as duplicate')
            
            return cleaned.capitalize()


        elif self.source.startswith('bpic2020'):
            cleaned = cleaned.replace('declaration submitted by employee', 'Employee submitted the declaration')
            cleaned = cleaned.replace('declaration approved by administration', 'Administration approved the declaration')
            cleaned = cleaned.replace('declaration final_approved by supervisor', 'Supervisor finally approved the declaration')
            cleaned = cleaned.replace('request payment', 'Request for payment was submitted')
            cleaned = cleaned.replace('payment handled', 'Payment was handled')
            cleaned = cleaned.replace('declaration approved by budget owner', 'Budget owner approved the declaration')
            cleaned = cleaned.replace('declaration approved by pre_approver', 'Pre-approver approved the declaration')
            cleaned = cleaned.replace('declaration rejected by administration', 'Administration rejected the declaration')
            cleaned = cleaned.replace('declaration rejected by employee', 'Employee rejected the declaration')
            cleaned = cleaned.replace('declaration rejected by supervisor', 'Supervisor rejected the declaration')
            cleaned = cleaned.replace('declaration rejected by budget owner', 'Budget owner rejected the declaration')
            cleaned = cleaned.replace('declaration rejected by pre_approver', 'Pre-approver rejected the declaration')
            cleaned = cleaned.replace('declaration rejected by missing', 'The system rejected the declaration')
            cleaned = cleaned.replace('declaration saved by employee', 'Employee saved the declaration')
            cleaned = cleaned.replace('other', 'Other action')
            return cleaned.capitalize()
        
        elif self.source.startswith('bpic2017'):
            cleaned = cleaned.replace('create offer', 'Create a new loan offer')
            cleaned = cleaned.replace('created', 'Offer was created')
            cleaned = cleaned.replace('sent (mail and online)', 'Offer sent via mail and online')
            cleaned = cleaned.replace('sent (online only)', 'Offer sent online only')
            cleaned = cleaned.replace('returned', 'Customer returned the offer')
            cleaned = cleaned.replace('accepted', 'Customer accepted the offer')
            cleaned = cleaned.replace('cancelled', 'Offer was cancelled')
            cleaned = cleaned.replace('refused', 'Customer refused the offer')
            
            return cleaned.capitalize()
        
        elif self.source.startswith('BPI12W'):
            # Dutch to English translations for BPI12W dataset
            cleaned = cleaned.replace('W_completeren aanvraag', 'Complete the application')
            cleaned = cleaned.replace('W_nabellen offertes', 'Call back for quotes')
            cleaned = cleaned.replace('W_valideren aanvraag', 'Validate the application')
            cleaned = cleaned.replace('W_afhandelen leads', 'Handle leads')
            cleaned = cleaned.replace('W_nabellen incomplete dossiers', 'Call back incomplete files')
            cleaned = cleaned.replace('W_beoordelen fraude', 'Assess fraud')
            

            
            return cleaned.capitalize()

        elif self.source.startswith('sepsis'):
            cleaned = cleaned.replace('er', 'ER').replace('iv', 'IV')
            return cleaned.capitalize()

        # default
        return cleaned.capitalize()


    def sequence_to_sentence_with_time(
        self,
        activities: List[str],
        timestamps: List[datetime],
        act_to_text: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Convert a sequence of activities with timestamps into a natural language sentence.

        Example output:
        "First, declaration submitted by employee. After 3.5 hours, declaration approved by supervisor."

        Args:
            activities (List[str]): Raw activity names.
            timestamps (List[datetime]): Per-step timestamps.
            act_to_text (Optional[Dict[str, str]]): Optional mapping raw->cleaned text.

        Returns:
            str: A sentence like "First, ..., After X hours, ...".
        """
        if act_to_text is None:
            act_to_text = {a: a for a in activities}  # fallback

        sentence = []
        for i, (act, ts) in enumerate(zip(activities, timestamps)):
            desc = act_to_text.get(act, act)
            if i == 0:
                sentence.append(f"First, {desc.lower()}.")
            else:
                delta = timestamps[i] - timestamps[i - 1]
                hours = round(delta.total_seconds() / 3600, 1)
                sentence.append(f"After {hours} hours, {desc.lower()}")

        return ' '.join(sentence)

    def encode_texts(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Generate text embeddings.

        Args:
            texts (Union[str, List[str]]): A single string or list of strings.
            batch_size (int): Batch size (SentenceTransformer).

        Returns:
            torch.Tensor: Embeddings of shape [B, D].
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if hasattr(self, 'tokenizer'):  # If using a transformer model
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu()

        else:  # SentenceTransformer
            embeddings = self.model.encode(
                texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            embeddings = torch.from_numpy(embeddings)
            return embeddings.to(self.device)

    # def process_sequence(self, 
    #                      sequence: List[str], 
    #                      act_to_text: Dict[str, str]) -> str:
    #     """
    #     Convert an activity sequence into natural language.
    #     """
    #     actions = []
    #     for i, act in enumerate(sequence):
    #         desc = act_to_text.get(act, act)
    #         if i == 0:
    #             actions.append(desc)
    #         elif i == 1:
    #             actions.append(f"then {desc}")
    #         else:
    #             actions.append(f"after that {desc}")
    #     return ', '.join(actions)
    def load_embeddings_dict(cache_path: str) -> Dict[str, torch.Tensor]:
        """
        Load embeddings saved in safetensors format.

        Args:
            cache_path (str): Base path (without extension) for safetensors and keys JSON.

        Returns:
            Dict[str, torch.Tensor]: Mapping from activity name to embedding tensor.
        """
        # Load tensor
        data = load_file(cache_path + ".safetensors")
        stacked_tensor = data['embeddings']  # [num_acts, embed_dim]

        # Load key mapping
        with open(cache_path + "_keys.json", 'r') as f:
            keys = json.load(f)

        return {k: stacked_tensor[i] for i, k in enumerate(keys)}
        


if  __name__ == '__main__':
    import pandas as pd
    from datetime import datetime


    df = pd.read_csv('C:/Users/Jingyi/Desktop/MA/data/BPIC20_processed.csv')  
    
    case_id = df['case_id'].unique()[0]
    case_df = df[df['case_id'] == case_id].sort_values('timestamp')
    
    activities = case_df['activity'].tolist()
    timestamps = case_df['timestamp'].tolist()


    processor = TextProcessor(source='bpic2020', model_name='prajjwal1/bert-medium')


    act_to_text = processor.clean_activity_names(activities)


    sentence = processor.sequence_to_sentence_with_time(activities, timestamps, act_to_text)
    print("\n📝 Generated Sentence:\n", sentence)


    embedding = processor.encode_texts(sentence)
    print("\n🔢 Embedding shape:", embedding.shape)  # [1, hidden_dim]