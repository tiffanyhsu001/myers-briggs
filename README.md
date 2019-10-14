# Myers-Briggs Personalities Classification

### Context:
The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis: <br />

Introversion (I) – Extroversion (E)
Intuition (N) – Sensing (S)
Thinking (T) – Feeling (F)
Judging (J) – Perceiving (P)

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
After the split, I checked for class imbalances, and there were heavy imbalances in the I,E and N,S subcategories. 
I also performed a chi-sq test for independence on the 6 different combinations of personality pairs (for example I_E and N_S, N_S and J_P, etc) and discovered that most of the subcategories are actually dependent on each other.


### Feature Engineering
In order to actually get some working variables for modelling, I created the following:
1. Frequency variables
  a. number of exclamation marks
  b. number of question marks
  c. number of ellipses
  d. words per comment
  e. links per comment
  f. number of nouns
  g. number of verbs
2. Sentiment analysis
  a. average AFINN score per comment
  b. average textblob polarity per comment
  c. average textblob subjectivity per comment
  

### Modeling 
After breaking the categories into four separate binary classification problems, the next step was to create classification models for each personality subcategoriy. To begin, I started with a simple model: logisitic regression


