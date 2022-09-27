import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.corpus import ChineseDailyNerCorpus
x_train,y_train = ChineseDailyNerCorpus.load_data('train')
x_valid,y_valid = ChineseDailyNerCorpus.load_data('validate')
x_test,y_test = ChineseDailyNerCorpus.load_data('test')
bert_embedding = BERTEmbedding('chinese_L-12_H-768_A-12',task=kashgari.labeling,sequence_length=100)
model = BiLSTM_CRF_Model(bert_embedding)
model.fit(x_train,y_train,x_validate=x_valid,y_validate=y_valid,epochs=200,max_batch_size=100)
