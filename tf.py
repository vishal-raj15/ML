import tensorflow as tf 
import tensorflow_datasets as tfds 

df = tfds.load(name='imdb_reviews')
print(df)