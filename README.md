Code relating to the study carried out for my master's thesis in Computer Science (Curriculum in Data Science)

<h2> Masked Autoregressive Flow  Models for Music Generation </h2>
Author: Massimiliano Sirgiovanni

Supervisor: Dott. Daniele Castellana

<hr>

The core of the thesis focuses on exploring the realm of Flow Based Models, aiming to adapt the Masked Autoregressive Flow (MAF) for the use case of music generation. This model has not been previously explored in this domain, in accordance with the knowledge currently available to us. The choice of MAF is guided by its promising ability to handle temporal data with strong dependencies between variables. Once the theoretical aspects related to this model are understood, it was decided to address three different approaches for creating an MAF model for music:
- MAF Univariate Shared;
- MAF Univariate DNDW;
- MAF Multivariate.

In **MAF Univariate Shared**, the univariate sequences related to each note are processed by the same MAF model, thus sharing its parameters.
<img src=>

In **MAF Univariate DNDW**, each sequence associated with a different note is processed by a different MAF with its own parameters.
<img src=>

In **MAF Multivariate**, where tracks are considered as multivariate sequences. With this approach, it is possible to capture relationships between different notes in different time steps.
<img src=>

<hr>
The work is divided into a series of files:

- accuracyMetrics --> contains all the methods that implement metrics for evaluating accuracy and generations;
- config --> contains the configurations of hyperparameters and settings used to perform the experiments;
- generationFunct --> contains all the functions useful for generating melodies;
- genreAnalysis --> contains methods for analyzing conditioning;
- initFolders --> initialization of useful directories;
- madeModel --> contains the class for defining the MADE model;
- mafModel --> contains the class for defining the MAF model;
- main --> contains the code to launch a text interface from which to operate on models;
- manageFiles --> contains useful methods for loading information and models from external files or saving information and models to external files;
- manageMemory --> contains methods to optimize used memory;
- manageMIDI --> contains methods to manage all operations related to MIDI files and piano rolls;
- maskedLinear --> contains the classes that implement the three variants of Masked Linear;
- modelSelection --> code to perform Model Selection;
- plot --> contains all the useful methods for creating plots;
- train --> contains all useful methods for training models;
- vaeModel --> contains the class for defining a simple VAE model;

**ATTENTION**: To be able to carry out the example developed for the thesis it is necessary to unzip the file --> ./savedObjects/dataset/2_bar/dataset_complete_program=0.zip
