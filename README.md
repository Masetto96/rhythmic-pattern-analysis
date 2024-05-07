# rythmic-pattern-analysis
## internship @ music technology group

#### The steps to carry out for this project are:
- Implement some rhythm descriptors (that will be integrated into Essentia).
- Compare existing rhythm descriptors for the task of rhythm pattern recognition.
- Ideally, propose improvements for the deep learning approach to reach the state of the art.


**computation_demo.ipynb:**  
   Illustrates how different parameters impact the computation of the scale transform magnitude (STM). It provides an overview of all the necessary steps involved in the computation.

**transformation_demo.ipynb:**  
   Evaluates the degree to which the feature maintains its invariance despite transformations applied to the signal.

**classification_demo.ipynb:**  
   Showcases the practical application of the feature for classification purposes utilizing the Groove MIDI dataset for drum fill style classification (e.g., funk, rock) and ballroom music datasets for musical dance genre classification. 

**helpers.py:**  
  Contains the implementation of the magnitude scale transformation.