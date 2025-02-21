import os
import numpy as np
import tensorflow as tf, tf_keras
import tensorflow_text as tftext
import tensorflow_datasets as tfds
import json
import collections
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_models as tfm
from tensorflow_models.nlp import layers
import tokenizers
import transformers
from tokenizers.normalizers import NFKC
import pandas as pd
nlp = tfm.nlp

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)


    
def build_vocab(tf_dataset): # reddit
    
    reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )
    text_numpy = tfds.as_numpy(text)
    content = text.map(lambda x: x["content"])
    #print(content)
    
    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        content.take(8192).batch(8192).prefetch(tf.data.AUTOTUNE),
        **bert_vocab_args
    )
    
    write_vocab_file('reddit_vocab.txt', pt_vocab)
    
def build_vocab_hugging(texts):
    print("starting vocab generation")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = tokenizers.normalizers.NFKC()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=8000,  # Small vocabulary size for demonstration (can be much larger)
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(texts, trainer)
    tokenizer.save("bpe_tokenizer.json")
def download_dump_dataset():
    tfds_dataset = tfds.load("reddit", split='train', as_supervised=True)
    data_list = [{"content": content.numpy().decode("utf-8"), "summary": summary.numpy().decode("utf-8")} for content, summary in tfds_dataset]
    # Convert to a Pandas DataFrame (optional)
    df = pd.DataFrame(data_list)
    df["content"].to_csv("reddit_content.txt", index=False, header=False)
    df["summary"].to_csv("reddit_summary.txt", index=False, header=False)
    # Print first rows
    print(df.head())

#content = text.map(lambda x: x["content"])
#print(text)
#files = os.listdir("/home/lorusso/tensorflow_datasets/reddit/1.0.0")
# Filter the files to get only those that start with "reddit"
#reddit_files = ["/home/lorusso/tensorflow_datasets/reddit/1.0.0/"+f for f in files if f.startswith("reddit")]
 

# Function to create vocabulary
#build_vocab_hugging(["reddit_summary.txt"])
#tokenizer = tokenizers.Tokenizer.from_file("bpe_tokenizer.json")
tokenizer = transformers.BertTokenizer("bpe_vocab_custom.txt")
#tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
# Access the vocabulary and save it to a file
#vocab = tokenizer.get_vocab()
#
#with open("bpe_vocab_custom.txt", "w") as vocab_file:
#    for token, idx in vocab.items():
#        vocab_file.write(f"{token}\n")


with open("reddit_content.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
# Strip the newline characters if you don't want them
content_list = [line.strip() for line in lines]
max_s = 512 # max([len(s) for s in content_list])
print(max_s)
print(content_list[2])
encoded_input = tokenizer(
        content_list[:10], 
    padding="max_length",
    truncation=True,
    max_length=max_s,
    return_tensors="tf",  # "pt" for PyTorch
    return_attention_mask=True,
    return_token_type_ids=True
)

input_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]
token_type_ids = encoded_input["token_type_ids"]

# Print results
#print("Input IDs:", input_ids.numpy())
#print("Attention Mask:", attention_mask.numpy())
#print("Token Type IDs:", token_type_ids.numpy())

#bert_tokenizer_params=dict(lower_case=True)

#reddit_tokenizer = tftext.BertTokenizer('reddit_vocab.txt', **bert_tokenizer_params)

#for ex in content.batch(3).take(1):
#    print(ex.numpy())
#
#test_inpt = reddit_tokenizer("Questa è una prova del tokenizer come dice chatgpt")
#print(test_inpt)


#tokenized_dataset = content.map(reddit_tokenizer.tokenize)

#token_batch = reddit_tokenizer.tokenize(next(iter(content.batch(8192))))

# Merge the word and word-piece axes -> (batch, tokens)
#tok_nump = tokenized_dataset.numpy()

#token_batch = token_batch.merge_dims(-2,-1)

#for ex in token_batch.to_list():
#    print(ex)
#
## Open and read the file
#with open("reddit_vocab.txt", "r", encoding="utf-8") as file:
#    lines = file.readlines()  # Reads all lines into a list
#
## Remove newline characters
#reddit_vocab = [line.strip() for line in lines]
##print(reddit_vocab)
#txt_tokens = tf.gather(reddit_vocab, token_batch)
## Join with spaces.
#print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))


cfg = {
    "vocab_size": 100,
    "hidden_size": 32,
    "num_layers": 3,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "activation": tfm.utils.activations.gelu,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "max_sequence_length": 16,
    "type_vocab_size": 2,
    "initializer": tf_keras.initializers.TruncatedNormal(stddev=0.02),
}
#bert_encoder = nlp.networks.BertEncoder(**cfg)
#
#def build_classifier(bert_encoder):
#    return nlp.models.BertClassifier(bert_encoder, num_classes=2)
#
#canonical_classifier_model = build_classifier(bert_encoder)
#canonical_classifier_model.summary()
#bert_encoder.summary()
#albert_encoder = nlp.networks.AlbertEncoder(**cfg)
#albert_encoder.summary()
#

#vocab_size,
#embedding_width=128,
#hidden_size=768,
#num_layers=12,
#num_attention_heads=12,
#max_sequence_length=512,
#type_vocab_size=16,
#intermediate_size=3072,
#activation=activations.gelu,
#dropout_rate=0.1,
#attention_dropout_rate=0.1,
#initializer=keras.initializers.TruncatedNormal(stddev=0.02),
#dict_outputs=False,

# word_ids = tf_keras.layers.Input(
#         shape=(None,), dtype=tf.int32, name='input_word_ids')
# mask = tf_keras.layers.Input(
#         shape=(None,), dtype=tf.int32, name='input_mask')
# type_ids = tf_keras.layers.Input(
#         shape=(None,), dtype=tf.int32, name='input_type_ids')
# 
# word_embedding = tfm.nlp.layers.OnDeviceEmbedding(vocab_size=100,embedding_width=128)(word_ids)
# type_embedding = tfm.nlp.layers.OnDeviceEmbedding(vocab_size=100,embedding_width=128)(type_ids)  # to recognize if a bunch of tokens are from the same sentence (using the [SEP] token in preprocessing)
# 
# position_embedding = tfm.nlp.layers.PositionEmbedding()(word_embedding)
# 
# embeddings = tf_keras.layers.Add()([word_embeddings, position_embeddings, type_embeddings])
# embeddings = (
#             tf_keras.layers.LayerNormalization(
#             name='embeddings/layer_norm',
#             axis=-1,
#             epsilon=1e-12,
#             dtype=tf.float32)(embeddings))
# 
# dropout_rate = 0.2
# embeddings = tf_keras.layers.Dropout(rate=dropout_rate)(embeddings)
# 
# #input_data = embeddings
# self_attention = tfm.nlp.layers.SelfAttentionMask()(embeddings,mask)
# 
# 
# 
# last_tr = tfm.nlp.layers.TransformerEncoderBlock()([embeddings, self_attention])
# 
# 
# model = tf_keras.Model(inputs = (word_ids,mask,type_ids),outputs = last_tr)
# 
# model.summary()
# 
class DivideLayer(tf_keras.layers.Layer):
    def __init__(self,n=3,**kwargs):
        super(DivideLayer, self).__init__()
        self.n = n
    def call(self,x):
        return tf.math.divide(x,self.n)




tf_keras.utils.register_keras_serializable(package='Text', name='LogEncoder')
class LogEncoder(tf_keras.Model):

    def __init__(self,
                 vocab_size,
                 embedding_width=128,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_sequence_length=512,
                 type_vocab_size=16,
                 intermediate_size=3072,
                 activation=tf_keras.activations.gelu,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer=tf_keras.initializers.TruncatedNormal(stddev=0.02),
                 dict_outputs=False,
                 **kwargs):
        activation = tf_keras.activations.get(activation)
        initializer = tf_keras.initializers.get(initializer)
    
        word_ids = tf_keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_word_ids')
        mask = tf_keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_mask')
        type_ids = tf_keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='input_type_ids')
  
        embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=tfm.utils.clone_initializer(initializer),
          name='word_embeddings')
        word_embeddings = embedding_layer(word_ids)
  
      # Always uses dynamic slicing for simplicity.
        position_embedding_layer = layers.PositionEmbedding(
            initializer=tfm.utils.clone_initializer(initializer),
            max_length=max_sequence_length,
            name='position_embedding')
        position_embeddings = position_embedding_layer(word_embeddings)
    
        type_embeddings = (
            layers.OnDeviceEmbedding(
                vocab_size=type_vocab_size,
                embedding_width=embedding_width,
                initializer=tfm.utils.clone_initializer(initializer),
                use_one_hot=True,
                name='type_embeddings')(type_ids))
  
        embeddings = tf_keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings])
        embeddings = (
            tf_keras.layers.LayerNormalization(
                name='embeddings/layer_norm',
                axis=-1,
                epsilon=1e-12,
                dtype=tf.float32)(embeddings))
        embeddings = (tf_keras.layers.Dropout(rate=dropout_rate)(embeddings))
        # We project the 'embedding' output to 'hidden_size' if it is not already
        # 'hidden_size'.
        

        # La trasformazione della dimensione di output viene fatta prima del transformer, da tenere in considerazione come è fatto ma forse meglio dopo?
        if embedding_width != hidden_size:
            embeddings = tf_keras.layers.EinsumDense(
              '...x,xy->...y',
              output_shape=hidden_size,
              bias_axes='y',
              kernel_initializer=tfm.utils.clone_initializer(initializer),
              name='embedding_projection')(
                  embeddings)
  
        data = embeddings
        attention_mask = layers.SelfAttentionMask()(data, mask)
        shared_layer = layers.TransformerEncoderBlock(
            num_attention_heads=num_attention_heads,
            inner_dim=intermediate_size,
            inner_activation=activation,
            output_dropout=dropout_rate,
            attention_dropout=attention_dropout_rate,
            kernel_initializer=tfm.utils.clone_initializer(initializer),
            name='transformer')
        encoder_outputs = []
        for _ in range(num_layers):
          data = shared_layer([data, attention_mask])
        #  encoder_outputs.append(data)
        


        encoder_out = tf_keras.layers.Dense(64)(data)
       # Applying a tf.slice op (through subscript notation) to a Keras tensor
       # like this will create a SliceOpLambda layer. This is better than a Lambda
       # layer with Python code, because that is fundamentally less portable.
#        first_token_tensor = data[:, 0, :]
#        cls_output = tf_keras.layers.Dense(
#            units=hidden_size,
#            activation='tanh',
#            kernel_initializer=tfm.utils.clone_initializer(initializer),
#            name='pooler_transform')(
#                first_token_tensor)
#        if dict_outputs:
        #  outputs = dict(
        #      sequence_output=data,
        #      encoder_outputs=encoder_outputs,
#       #       pooled_output=cls_output,
        #  )

        decoder_input = tf_keras.layers.Dense(64)(encoder_out)
        
        data_inv = tf_keras.layers.Dense(hidden_size)(decoder_input)
        #data = (tf_keras.layers.Dropout(rate=dropout_rate)(data))
        shared_layer = layers.TransformerEncoderBlock(
            num_attention_heads=num_attention_heads,
            inner_dim=intermediate_size,
            inner_activation=activation,
            output_dropout=dropout_rate,
            attention_dropout=attention_dropout_rate,
            kernel_initializer=tfm.utils.clone_initializer(initializer),
            name='transformer_inverse')
        attention_mask = layers.SelfAttentionMask()(data_inv, mask)
        for _ in range(num_layers):
            data_inv = shared_layer([data_inv, attention_mask])

        if embedding_width != hidden_size:
          embeddings = tf_keras.layers.EinsumDense(
              '...x,xy->...y',
              output_shape=embedding_width,
              bias_axes='y',
              kernel_initializer=tfm.utils.clone_initializer(initializer),
              name='embedding_projection_inverse')(
                  data_inv)

        embeddings = tf_keras.layers.Dropout(rate=dropout_rate)(embeddings)
        
        split_embeddings =  DivideLayer(n=3.0)(embeddings)
        output_word_ids = tf_keras.layers.Dense(vocab_size, activation='softmax')(split_embeddings)
        output_type_ids = tf_keras.layers.Dense(type_vocab_size, activation='softmax')(split_embeddings)



        decoder_out = output_word_ids


#        else:
#          outputs = [data, cls_output]
  
      # b/164516224
      # Once we've created the network using the Functional API, we call
      # super().__init__ as though we were invoking the Functional API Model
      # constructor, resulting in this object having all the properties of a model
      # created using the Functional API. Once super().__init__ is called, we
      # can assign attributes to `self` - note that all `self` assignments are
      # below this line.
        super().__init__(
            inputs=(word_ids, mask, type_ids), outputs=decoder_out, **kwargs)
        config_dict = {
            'vocab_size': vocab_size,
            'embedding_width': embedding_width,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'max_sequence_length': max_sequence_length,
            'type_vocab_size': type_vocab_size,
            'intermediate_size': intermediate_size,
            'activation': tf_keras.activations.serialize(activation),
            'dropout_rate': dropout_rate,
            'attention_dropout_rate': attention_dropout_rate,
            'initializer': tf_keras.initializers.serialize(initializer),
        }
  
      # We are storing the config dict as a namedtuple here to ensure checkpoint
      # compatibility with an earlier version of this model which did not track
      # the config dict attribute. TF does not track immutable attrs which
      # do not contain Trackables, so by creating a config namedtuple instead of
      # a dict we avoid tracking it.
        config_cls = collections.namedtuple('Config', config_dict.keys())
        self._config = config_cls(**config_dict)
        self._embedding_layer = embedding_layer
        self._position_embedding_layer = position_embedding_layer
  
    def get_embedding_table(self):
      return self._embedding_layer.embeddings
  
    def get_config(self):
      return dict(self._config._asdict())
  
    @classmethod
    def from_config(cls, config):
      return cls(**config)

#albert = tfm.nlp.networks.AlbertEncoder(100)
#albert.summary()
model = LogEncoder(vocab_size=8128, max_sequence_length = max_s) # per ora messo manualmente da 8000 che dovrebbe essere, inserire modo automatico per misurare il vocabolario
model.summary()

############ DEFINE LOSS resolve problem with sliding of the loss instead of all 512 outputs at once

def reco_loss(y_true, y_pred):
    # Categorical Crossentropy for the first output
    loss_id = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    #assert tf.reduce_all((y_true >= 0) & (y_true < 8000))
    # Categorical Crossentropy for the second output
    #loss_type = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true[1], y_pred[1])
    
    # Combine the losses (e.g., sum or average)
    total_loss = loss_id #+ loss_type  # You can use any other combination such as (loss_1 + loss_2) / 2
    
    return total_loss

model.compile(optimizer = tf_keras.optimizers.Adam(learning_rate=3E-3,clipnorm=1.0),loss=reco_loss)
model.fit(x = (input_ids,attention_mask,token_type_ids),y = input_ids, epochs=1, batch_size=1)

encoded_input = tokenizer(
        content_list[705:707],
    padding="max_length",
    truncation=True,
    max_length=max_s,
    return_tensors="tf",  # "pt" for PyTorch
    return_attention_mask=True,
    return_token_type_ids=True
)

input_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]
token_type_ids = encoded_input["token_type_ids"]

print(content_list[705])
test_out = model.predict((input_ids, attention_mask,token_type_ids))
max_out= np.max(test_out[0],axis=1)
tok_idxs = []
for i in test_out[0]:
    print(len(i),i)
    tok_idxs.append(np.argmax(i))
    print(np.argsort(i)[-2])
    print(np.sort(i)[::-1])
print(tok_idxs)
cat_out = np.argsort(test_out[0],axis=0)[-2]
print(test_out.shape)
print(test_out[0].shape)
print(test_out[0][0].shape)
print(max_out.shape)
print(cat_out.shape)
print(cat_out)
print(tokenizer.decode(tok_idxs,skip_special_tokens=True))


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)
  
        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)
  
        ## Create the signatures for export:   
  
        # Include a tokenize signature for a batch of strings. 
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))
  
        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
  
        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
  
        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()
  
    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2,-1)
        enc = add_start_end(enc)
        return enc
  
    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)
  
    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)
  
    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
  
    @tf.function
    def get_vocab_path(self):
        return self._vocab_path
  
    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

