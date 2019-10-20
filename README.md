# Myers-Briggs Personalities Classification

### Context:
The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis: <br />

Introversion (I) – Extroversion (E) <br />
Intuition (N) – Sensing (S) <br />
Thinking (T) – Feeling (F) <br />
Judging (J) – Perceiving (P) <br />

### Data:
Each row of data contains: 
  - Posts: a section of each of the last 50 things they have posted (Each entry separated by "|||" (3 pipe characters))
  - Type: personality type (i.e. INFJ)

### Objective:
Predicting personality types based off 50 forum postings (text) from users online.

### Set up/ Initial Analysis:
Upon initial investigation, I realized that the data had zero numeric features & no missing values. However, each row had exactly 50 posts. To get a better idea of the actual data, I split the personality types into four separate subcategories:
  1. Introversion (I) – Extroversion (E)
  2. Intuition (N) – Sensing (S)
  3. Thinking (T) – Feeling (F)
  4. Judging (J) – Perceiving (P) <br />
<br />
After the split, I checked for class imbalances, and there were heavy imbalances in the I,E and N,S subcategories. 
I also performed a chi-sq test for independence on the 6 different combinations of personality pairs (for example I_E and N_S, N_S and J_P, etc) and discovered that most of the subcategories are actually dependent on each other.


### Feature Engineering
In order to actually get some working variables for modelling, I created the following:
1. Frequency variables <br />
  a. number of exclamation marks <br />
  b. number of question marks <br />
  c. number of ellipses <br />
  d. words per comment <br />
  e. links per comment <br />
  f. number of nouns <br />
  g. number of verbs <br />
2. Sentiment analysis <br />
  a. average AFINN score per comment <br />
  b. average textblob polarity per comment <br />
  c. average textblob subjectivity per comment <br />
  

### Modeling 
After breaking the categories into four separate binary classification problems, the next step was to create classification models for each personality subcategoriy. For validation, I broke the entire dataset in a training & testing set (70/30 split). To begin, I started with a simple model: logisitic regression. 

#### Neural Net
To capture nonlinear relationships, I also built a basic neural network model. To implement the model, I used the keras package and built a model with 2 hidden layers. I included 6 nodes in each hidden layer following the general rule of thumb:  <br />

num of nodes = mean(input layer + output layer) <br />
<br />
as I had 11 input layers (variables) and 1 output layer. Additionally, I added regularizers and a dropout in each hidden layer to reduce overfitting. In my output layer, I included a sigmoid activation function because my model is performing binary classification. When compiling the model, I chose to go with binary_crossentropy loss because again, the model is binary classification. I also went with rmsprop optimizer as it performed better than adam or sgd.

#### Random Forest
For my random forest modeling, I used the sklearn's random forest classifier. To tune the hyperparameters, I utilized grid search cross validation with CV = 5 and f1 scoring due to the nature of the imbalanced data. All scoring was ultimately determined by the initial hold-out test set. 

### Evaluation, Metrics
For all models, the metrics I focused on were accuracy and f1. I kept accuracy as a metric because of its straightforward interpretability. On the other hand, since the data is imbalanced, I chose to also include f1 because it also accounts for penalties from imbalanced data.

### Results

#### Logistic Regression
| Scoring     | E vs I | I vs S | F vs T | J vs P | Total |
|-------------|--------|--------|--------|--------|-------|
| Accuracy    | 0.76   | 0.53   | 0.66   | 0.61   | 0.162 |
| Weighted F1 | 0.66   | 0.60   | 0.66   | 0.46   | 0.120 |

#### Neural Network
| Scoring     | E vs I | I vs S | F vs T | J vs P | Total |
|-------------|--------|--------|--------|--------|-------|
| Accuracy    | 0.63   | 0.61   | 0.66   | 0.52   | 0.132 |
| Weighted F1 | 0.65   | 0.67   | 0.67   | 0.52   | 0.152 |

#### Random Forest 
| Scoring     | E vs I | I vs S | F vs T | J vs P | Total |
|-------------|--------|--------|--------|--------|-------|
| Accuracy    | 0.75   | 0.84   | 0.64   | 0.52   | 0.209 |
| Weighted F1 | 0.67   | 0.80   | 0.64   | 0.52   | 0.178 |

### Conclusion 
After evaluating each of the models, the random forest classification model performed best, with the highest overall 'total' scoring accuracy of 0.209. Similarly, the random forest model was had the most balanced performance with an f1 of 0.178. Logistic regression followed in second in accuracy, but was much lower than the neural network model in f1. Overall, for these specific dataset, I would recommend using the random forest classifier because of its robustness against outliers and preferred nature to imbalanced datasets.

