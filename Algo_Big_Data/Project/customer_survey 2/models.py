import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC

class Models:
    def __init__(self, train, test):
        self. train = train
        self.test = test

    def plot_confusion_matrix(self, cm, classes=None, title=None, model=None):
        if classes is not None:
            sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0, vmax=1, annot=True, annot_kws={'size':40})
        else:
            sns.heatmap(cm, vmin=0, vmax=1)
        plt.title(title+" "+model)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        path = model.lower().replace(" ","_")
        plt.savefig("plots_images/"+path+".png")
        plt.close()

    def logistic_regression(self):
        lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
        lrModel = lr.fit(self.train)
        #Plot Beta Coefficients
        beta = np.sort(lrModel.coefficients)
        plt.plot(beta)
        plt.ylabel('Beta Coefficients')
        plt.draw()
        plt.savefig("plots_images/beta_coefficient.png")
        plt.close()
        #plt.show()
        
        trainingSummary = lrModel.summary
        roc = trainingSummary.roc.toPandas()
        plt.plot(roc['FPR'],roc['TPR'])
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.draw()
        plt.savefig("plots_images/roc_curve.png")
        #plt.show()
        print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
        #Plot Precision and Recall
        pr = trainingSummary.pr.toPandas()
        plt.plot(pr['recall'],pr['precision'])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.draw()
        plt.savefig("plots_images/precision_recall.png")
        plt.close()
        #Predictions
        predictions = lrModel.transform(self.test)
        
        evaluator = BinaryClassificationEvaluator()
        print('Test Area Under ROC', evaluator.evaluate(predictions))
        predictions_display = predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
        print(predictions_display)

        #Plot Confudion Matrix
        y_pred = predictions.select("prediction").collect()
        y_label = predictions.select("label").collect()

        cm = confusion_matrix(y_label, y_pred)

        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        classes = np.unique(y_label)
        self.plot_confusion_matrix(cm_norm, classes, title="Confusion Matrix for Build Model", model="Logistic regression")
    
    def decision_tree(self):
        dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
        dtModel = dt.fit(self.train)
        predictions_dt = dtModel.transform(self.test)
        predictions_dt_display = predictions_dt.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
        print(predictions_dt_display)
        evaluator = BinaryClassificationEvaluator()
        print("Test Area Under ROC: " + str(evaluator.evaluate(predictions_dt, {evaluator.metricName: "areaUnderROC"})))
        
        #Plot Confusion Matrix
        y_pred = predictions_dt.select("prediction").collect()
        y_label = predictions_dt.select("label").collect()

        cm = confusion_matrix(y_label, y_pred)

        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        classes = np.unique(y_label)
        self.plot_confusion_matrix(cm_norm, classes, title="Confusion Matrix for Build Model", model="Decision Tree")
    
    def random_forest(self):
        rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
        rfModel = rf.fit(self.train)
        predictions_rf = rfModel.transform(self.test)
        predictions_rf_display = predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
        print(predictions_rf_display)
        evaluator = BinaryClassificationEvaluator()
        print("Test Area Under ROC: " + str(evaluator.evaluate(predictions_rf, {evaluator.metricName: "areaUnderROC"})))
        
        #Plot Confusion Matrix
        y_pred = predictions_rf.select("prediction").collect()
        y_label = predictions_rf.select("label").collect()

        cm = confusion_matrix(y_label, y_pred)

        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        classes = np.unique(y_label)
        self.plot_confusion_matrix(cm_norm, classes, title="Confusion Matrix for Build Model", model="Random Forest")
    
    def gradient_boosting(self):
        gbt = GBTClassifier(maxIter=10)
        gbtModel = gbt.fit(self.train)
        predictions_gb = gbtModel.transform(self.test)
        predictions_gb_display = predictions_gb.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
        print(predictions_gb_display)
        evaluator = BinaryClassificationEvaluator()
        print("Test Area Under ROC: " + str(evaluator.evaluate(predictions_gb, {evaluator.metricName: "areaUnderROC"})))

        #Plot Confusion Matrix
        y_pred = predictions_gb.select("prediction").collect()
        y_label = predictions_gb.select("label").collect()

        cm = confusion_matrix(y_label, y_pred)

        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        classes = np.unique(y_label)
        self.plot_confusion_matrix(cm_norm, classes, title="Confusion Matrix for Build Model", model="Gradient Boosting")
    
    def linear_support_vector_machine(self):
        lsvc = LinearSVC(maxIter=10, regParam=0.1)
        # Fit the model
        lsvcModel = lsvc.fit(self.train)
        # Print the coefficients and intercept for linearsSVC

        predictions_svm = lsvcModel.transform(self.test)
        predictions_svm_display = predictions_svm.select('age', 'job', 'label', 'rawPrediction', 'prediction').show(10)
        print(predictions_svm_display)
        #Plot Confusion Matrix
        y_pred = predictions_svm.select("prediction").collect()
        y_label = predictions_svm.select("label").collect()
        cm = confusion_matrix(y_label, y_pred)
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        classes = np.unique(y_label)
        self.plot_confusion_matrix(cm_norm, classes, title="Confusion Matrix for Build Model", model="Support Vector Machine")
