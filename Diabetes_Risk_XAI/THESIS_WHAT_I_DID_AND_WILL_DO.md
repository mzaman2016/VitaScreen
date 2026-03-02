# What I Have Done and What I Will Do – Simple English Explanation

---

## What I Have Done (Completed)

### Introduction
I wrote the introduction for my thesis. I explained that diabetes is a big health problem and that we need to predict who is at risk early. I said that many AI models are like black boxes—we cannot see why they make a decision. So I used Explainable AI (XAI) with Grad-CAM to show which parts of the data the model looks at when it predicts diabetes. I also wrote the problem statement (why we need explainability), motivation (why trust in AI matters), and my main contributions (the pipeline I built, the feature names and scales I added, the 84% accuracy, and the GitHub link).

### Background Knowledge
I wrote the background chapter. I explained five things in simple words:
1. **IGTD** – A method that turns table data (numbers in rows and columns) into small grayscale images. We use a 5×3 grid for 15 features.
2. **CNN** – The deep learning model we use. It has 4 convolutional blocks, and we follow Shenghao Wang’s design.
3. **Grad-CAM** – The explainability method. It shows which parts of the image the model focused on. Red = important, blue = not important.
4. **ENN** – A step that removes noisy samples from the training data using k=3 neighbors.
5. **CDC Dataset** – The health data we use. It has 15 features like HighBP, BMI, Smoker, etc., and a label (diabetic or non-diabetic).

---

## What I Will Do (Remaining Chapters)

### Literature Review
**What I will do:** I will read and summarize papers about IGTD (Zhu et al.), Grad-CAM (Selvaraju et al.), and diabetes prediction with CNNs (Wang et al., VitaScreen). I will compare our work with these papers and say how our approach is similar or different.

### Proposed Methodology
**What I will do:** I will describe the full pipeline step by step: (1) Load CDC data, (2) Pick 15 features, (3) Split 80:20 train/test, (4) Apply ENN on the training set, (5) Run IGTD to create images, (6) Train the CNN, (7) Apply Grad-CAM and save heatmaps with feature names and scales. I will also explain how we add feature names and color scales to the Grad-CAM images.

### Implementation
**What I will do:** I will explain the code. I will describe the base pipeline (diabetes_prediction_pipeline.py) and the XAI pipeline (gradcam_explainability.py). I will show the main functions and how they connect. I will also give the GitHub link and how to run the code.

### Result Discussion
**What I will do:** I will show the results table (Accuracy 0.84, Precision 0.38, Recall 0.50, F1-Score 0.43). I will add the 8 Grad-CAM sample figures (gradcam_sample_1.png to gradcam_sample_8.png). I will discuss what the heatmaps show—for example, that Veggies, AnyHealthcare, and CholCheck are often important. I will also explain why the model does better on non-diabetic cases (because there are more of them in the data).

### Conclusion
**What I will do:** I will summarize the whole project. I will say that we built an explainable diabetes prediction pipeline using IGTD and Grad-CAM, achieved 84% accuracy, and added feature names and scales so people can understand the model’s decisions. I will mention the GitHub link again.

### Limitations and Future Work
**What I will do:**  
- **Limitations:** I will say that (1) there are more non-diabetic than diabetic samples, so the model is better at non-diabetic cases; (2) Grad-CAM shows image regions, not direct feature importance; (3) we used 500 samples, so results may change with more data.  
- **Future Work:** I will say we could (1) use SHAP or LIME for feature-level explanations; (2) use the full CDC dataset or the cdcNormalDiabeticFE1_20RFFSQ.csv file; (3) put Grad-CAM into a web app; (4) try NCTD instead of IGTD.
