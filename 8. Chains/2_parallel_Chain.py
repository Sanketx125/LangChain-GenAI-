from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# for parallel chian
from langchain.schema.runnable import RunnableParallel

load_dotenv()


model1 = ChatGoogleGenerativeAI(model= 'gemini-2.5-pro-preview-03-25')

model2 = ChatGoogleGenerativeAI(model= 'gemini-1.5-pro')


templet1 = PromptTemplate(
    template= "Generate short & simple note from following text \n {text}",
    input_variables= ['text'],

)

templet2 = PromptTemplate(
    template='Generate 5 short question answer from the following text, \n {text}',
    input_variables= ['text']
)


templet3 = PromptTemplate(
    template= 'merge the provided notes and quez into single document.\n notes -> {notes} and quez -> {quez}',
    input_variables= ['notes', 'quez']
)

parser = StrOutputParser()


# Parallel Chains

parallel_chain = RunnableParallel(
    {
        'notes': templet1 | model1 | parser,
        'quez' : templet2 | model2 | parser,
    }
)

# merge Chain
merge_chain = templet3 | model1 | parser


# final Chain
Chain = parallel_chain | merge_chain

text = '''
1.4. Support Vector Machines
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

1.4.1. Classification
SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset.


X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y)
SVC(decision_function_shape='ovo')
dec = clf.decision_function([[1]])
dec.shape[1] # 6 classes: 4*3/2 = 6
6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
4
On the other hand, LinearSVC implements “one-vs-the-rest” multi-class strategy, thus training n_classes models.

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
LinearSVC()
dec = lin_clf.decision_function([[1]])
dec.shape[1]
4
See Mathematical formulation for a complete description of the decision function.

Examples

Plot different SVM classifiers in the iris dataset

1.4.1.2. Scores and probabilities
The decision_function method of SVC and NuSVC gives per-class scores for each sample (or a single score per sample in the binary case). When the constructor option probability is set to True, class membership probability estimates (from the methods predict_proba and predict_log_proba) are enabled. In the binary case, the probabilities are calibrated using Platt scaling [9]: logistic regression on the SVM’s scores, fit by an additional cross-validation on the training data. In the multiclass case, this is extended as per [10].

Note

The same probability calibration procedure is available for all estimators via the CalibratedClassifierCV (see Probability calibration). In the case of SVC and NuSVC, this procedure is builtin in libsvm which is used under the hood, so it does not rely on scikit-learn’s CalibratedClassifierCV.

The cross-validation involved in Platt scaling is an expensive operation for large datasets. In addition, the probability estimates may be inconsistent with the scores:

the “argmax” of the scores may not be the argmax of the probabilities

in binary classification, a sample may be labeled by predict as belonging to the positive class even if the output of predict_proba is less than 0.5; and similarly, it could be labeled as negative even if the output of predict_proba is more than 0.5.
'''

result = Chain.invoke({'text': text})

print(result)