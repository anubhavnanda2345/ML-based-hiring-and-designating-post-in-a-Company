#First let's define a function that'll process the random skilss of candidates
#that we want to predict whether to hire them or not
variable = []
def skill_process(skills):
    review = re.sub('[^a-zA-Z]', ' ', str(skills[0]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    variable.append(review)
    return variable
   
#Import all the important packages
import numpy as np
import pandas as pd
dataset = pd.read_excel('sample_data_v4.xlsx')

#Import the rows containing the skills of different candidates to our variable X
X = dataset.iloc[:,2]

#Import NLTK for Natural Language Processing that we'll be using
import re
import nltk

#Define the skills that the company requires in each post in a list
Web_Developer = ['HTML, CSS, Bootstrap, AJAX, JavaScript, PHP, MySQL, REST API, AWS/GCP, Git/GitHub, Python, Linux, JSON, Authentication/Authorization (JWT)']
FullStack_Developer= ['HTML, CSS, MongoDB, Express, React.js, Node.js, ES6, Redux, JavaScript, AWS/GCP, Python, Linux, JSON, Authentication/Authorization (JWT)']
ML_Developer= ['Python, TensorFlow, PyTorch, NumPy, PyPI, Sci-kit learn, Statistical Modelling, Machine Learning, Deep Learning, SQL, JSON, AWS/GCP/Heroku, Flask']

#Now we perform stemming in our predefined skills and save them to corpus1,2&3 respectively
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
corpus1 = []
corpus2 = []
corpus3= []

#WEB DEVELOPER & save to corpus1
review = re.sub('[^a-zA-Z]', ' ', str(Web_Developer[0]))
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus1.append(review)

#FULL STACK DEVELOPER & save to corpus2
review = re.sub('[^a-zA-Z]', ' ', str(FullStack_Developer[0]))
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus2.append(review)

#ML DEVELOPER & save to corpus3
review = re.sub('[^a-zA-Z]', ' ', str(ML_Developer[0]))
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus3.append(review)


#Now we perform stemming in X and store the stemmed words into variable corpus
for i in range(0,2168):
   
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['Other skills'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
y =[]
y1=[]

#Now we find whether the skills of any candidate is a subsed of skills given in
#Web Developer,FullStack Developer & ML developer
#If the answer is yes we save the correspond post in same index no. of X in variable y
#If the answer is no  we save the 'Not hired' in same index no. of X in variable y

for i in range(0,2168):
    a = set(corpus1[0]).issubset(set(corpus[i]))
    b = set(corpus2[0]).issubset(set(corpus[i]))
    c = set(corpus3[0]).issubset(set(corpus[i]))
    if a == 1:
        y1 = ['Web Developer']
       
    elif b==1 :
        y1 = ['Full Stack developer']
       
    elif c == 1:
        y1 = ['ML Developer']

    else :
        y1 = ['Not hired']
    y.append(y1)    


#Now we have our output i.e variable y ready for Machine Learning

   
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 25000)
X = cv.fit_transform(corpus).toarray()

## using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier use any other classifier if u wish
classifier = DecisionTreeClassifier()

# training the model on X_train and y_train
classifier.fit(X_train,y_train)

# predict on X_test
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Now Let's predict on a random employee with random skill applying for a random position
skills = ['AJAX']
skills_temp = skills

skills = skill_process(skills)
skills = cv.fit_transform(skills).toarray()
skills = np.resize(skills,[1,307])

y_pred1 = classifier.predict(skills)

print('Enter the position you want to apply e.g. WD for Web Developer,FD for Full Stack Developer,ML for ML Developer')
s = str(input())

#Now  if the candidate is not hired, his skill difference will be printed
#If hired a greetings message will pop-up

if y_pred1 == ['Not hired']:
    if s == ['WD']:
        print('Skills youre lacking are as follows')
        k=str(Web_Developer).replace(str(skills_temp),' ')
        print(k)
   
    elif s ==  ['FD']:
        k=str(FullStack_Developer).replace(str(skills_temp),' ')
        print(k)
       
    elif s== ['ML']:
        k=str(ML_Developer).replace(str(skills_temp),' ')
        print(k)
else:
     print('Congratulations you are hired')
