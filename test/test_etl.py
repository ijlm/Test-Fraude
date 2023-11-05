import pytest
import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/../src')
import train_model

def test_evaluate_models():   
   accuracy,precision,recall,f1_score,roc_auc,matrix=train_model.evaluate_models([1,0,1,0,1,1],[1,1,1,1,0,1])
   assert accuracy==0.5
   assert precision==0.6
   assert recall==0.75
   assert f1_score>0.5
   assert roc_auc>0.2

def test_definir_corte_optimo():   
   threshold=train_model.definir_corte_optimo([0.5,0.5,0.3,0.1,0.9,1],[0.5,0.5,0.3,0.2,0.5,1], 1, 1, 9,0)   
   assert threshold>=0.0
