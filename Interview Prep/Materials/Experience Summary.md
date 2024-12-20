
- Hello! My name is Ayushe Gangal. I’ll be graduating with a Master’s in Computer Science (Data Science Concentration) from the University of Massachusetts Amherst at the end of this month.
    
- I’ve taken advanced courses in Data Science, Machine Learning, Statistics, Computational Statistics, Data Visualization, Deep Learning and Software Engineering. 
    
- Experience with AWS S3, AWS Redshift, Amazon SageMaker, PySpark, Data bricks
    
- Profile align more with Data Science Associate (R0036212)
    

<mark style="background: #D2B3FFA6;">Waters Data Engineer</mark>

- Currently, I’m working as a Data Engineer at Waters Corporation, which is a Scientific instrument/ Biotechnology Pharmaceutical company based in Milford, Massachusetts.
    
- I use machine learning principles to develop tools for interactive visualization and data processing in the domain of mass spectrometry. I mostly use supervised and unsupervised learning methodologies, and neural networks for imaging and text-based data.
    
- Have worked with large amounts of structured, unstructured, raw text data, and have worked with different file formats like csv, txt, apex, parquet, pickle, raw, numpy etc. jpeg, png, tiff, for images.
    
- Libraries I use: TensorFlow, Keras, Pytorch, Scikit Image, OpenCV, sci-pi, Plotly, Matplotlib, Bokeh, Seaborn, numpy, pandas
    
- Working solely on a Spatial multi-modal data [[Registration]] application, to register an optical-image to its mass spectrometry analyte file. Computer Vision, heavy image processing, data pre-processing and data-manipulation. Experience with feature engineering, creating composite features from raw features.
    
- Company Data-thon: “ Quadrupole assembly geometry assessment by Machine Learning” Given an open-ended problem, performed exploratory data analysis, created meaningful visualizations, used statistical computing to find correlations, performed feature engineering and feature selection. Which aspects of assembly geometry have a higher impact on the performance.
    
	- The data was high-dimensional and wide. Had ~150 sample data points with 200 features.
    
	- Converted the multi-class classification to a Binary-classification problem.
    
	- Performed feature engineering to create composite and more informative 97 “derived” features from the 200 “raw” features. Then performed sparser feature selection by Training an XGBoost model with heavy regularization which ultimately drove the number down to 27 features.
    
	- Result: 81.6% on a held-out test set (31/38 correct).
    
	- Won Honorable Mention in the Most Innovative Project category.
    
- Created important visualizations and extracted information from a 5-D unintelligible sparse multi-indexed (hierarchical) Ion Mobility & Imaging mass spectrometry data. 
    

<mark style="background: #D2B3FFA6;">Waters Software Engineer</mark>

- I’ve also worked as a Software Engineer at Waters Corporation where I worked with the Data Trending team, which worked on creating a Visualization Dashboard for the Scientists who worked with the Color Chromatography dataset from the company’s legacy software.
    
- I was in the team which migrated the client’s sensitive data from the company’s legacy software to Amazon S3 to Amazon Redshift. I utilized Entity Framework and SQL for this purpose, and created a local database by integrating SQLite and C# to keep track of the previous states of the chromatogram data. 
    
- Have also worked with Parquet data formats:
	- organized into row groups, which are divided into columns. 
	- Each column is stored as a separate block, and these blocks are compressed independently to achieve high compression ratios. 
	- This columnar layout makes it easy to skip over irrelevant data when performing queries, which can significantly improve query performance.
	- Parquet also supports schema evolution, which means that changes to the data structure can be made without having to rewrite the entire dataset. This makes it easy to add new columns or modify existing ones over time.
    
- Followed Agile Practices.
    

<mark style="background: #D2B3FFA6;">EleNa: Elevation based Navigation Application</mark>

- Graph Neural Networks VS Dijkstra: Worked on a elevation based navigation application which finds the best route with the least elevation from point A to B. Did a comparative study between GNN and Dijkstra for simple (non-elevation based) and elevation-based navigation application, and found that Graph Neural Networks (GNNs) are not the most efficient for elevation-based navigation for several reasons:
	- First, elevation data can be highly variable and noisy, making it difficult to represent and learn effectively with a GNN. 
	- Second, in some cases, elevation data may not be as important as other factors such as terrain, obstacles, and route availability in determining the optimal path. 
	- Third, GNNs are typically used for graph-structured data, and while elevation data can be represented as a graph, other techniques such as grid-based representations may be more efficient for certain types of navigation tasks.
	- Finally, GNNs require significant computational resources and may not be well-suited for real-time navigation applications that require fast response times.
    
-   Read up: [https://github.com/muditchaudhary/EleNA](https://github.com/muditchaudhary/EleNA)
    

<mark style="background: #D2B3FFA6;">Machine Learning Research Experience</mark>

- In the past, I have worked as Research Assistant in the Machine Learning Lab at my undergrad university for two years and have published 6 research articles in International Journals of repute, have presented at International Conferences, have written two book chapters for IGI Global publications, and have 2 patents under my name. I’ve majorly worked with Machine Learning algorithms in the predictive modeling, diagnostic analytics, and detection.
    
	- Patent I’m most proud of “Subway Train Seat Occupancy Detection System using Computer Vision”: because it was my first end-to-end project, from the formulation of the major idea, creation of data pipelines to filing for the patent.

		- Summary: Applied real-time occupancy monitoring for subway train’s seats using Computer Vision. We have also detected if someone is breaking the social distancing norms in the train and sitting on the prohibited seats or not. Image-masking and detection-free human instance segmentation has been employed, so as to efficiently segregate the vacant seats from the crowd. An application/ user interface has also been created for the users, which will inform them about the location of the vacant seats in the train. The user interface cogently displays the exact number of seats, which have been color-coded to enhance the users’ understanding of the application. Accuracy, f1-score, precision and recall used.
    
		- Model Creation: A miniature model of the subway train was constructed by following the scaled dimensions of the life-size subway train (metro). The interior and exterior of the train was kept intact, with windows, poles, seats at the exact scaled position and dimensions.
    
		- Dataset: Since a generalized dataset could not be used for this specific task, we used the miniature model to collect our own data. We used miniature human figures as passengers, which were detected as humans by 99.8% accuracy. Images from 4 angles, such that covering each and every seat in the coach, were captured.
    
		- Advantage: We were able to do all this and still our results scale to real-life usage as the environments we are dealing with are fixed. The position of CCTV cameras are fixed, the positions and dimensions of the seats and the coach are fixed as well. Only variable are the passengers, therefore, it is easy to generate a mask for these images from the 4 angles and detect seats and humans.
    
		- Methodology: Divided into 2 phases: initialization and detection.
			- Initialization calculates the min ROI bounding boxes and converts 2D points to 3D, to obtain ROI coordinates. It also calculates the passengers’ Center of Mass (calculated via averaging every human body pixel), width and length calculation, which is used to draw ellipses to mark humans.
			- Detection uses the min ROI co-ordinates and ellipses to detect vacant seats.
		If the COM lies within the ellipse, the seat is not vacant. Else , it is. 
		-   Results: 0.998 accuracy, 1 precision, 0.996 recall, 0.998 f1-score
    

	- Journal article: WisdomNet: Prognosis of COVID-19 with Slender Prospect of False Negative Cases and Vaticinating the Probability of Maturation to ARDS using Posteroanterior Chest X-Rays 
    
		- What we did: Utilized the concept of Wisdom of Crowds using a two-layered convolutional Neural Network (CNN), which takes chest x-ray images as input. Both layers of the proposed neural network consist of 80 neural networks each (obtained by experimental procedures). The network not only pinpoints the presence of COVID-19, but also gives the probability of the disease maturing into Acute Respiratory Distress Syndrome (ARDS). Thus, predicting the progression of the disease in the COVID-19 positive patients. The network also slender the occurrences of false negative cases by employing a high threshold value, thus aids in curbing the spread of the disease and gives an accuracy of 100% for successfully predicting COVID-19 among the chest x-rays of patients affected with COVID-19, bacterial and viral pneumonia. The threshold has been set to 50% to classify a posteroanterior x-rays as COVID-19 positive, but it has been set to 70% to classify the x-rays as COVID-19 negative. Thus, if and only if the model is more than 70% sure that a case is negative, then only it will be classified as Covid negative.
    
		- Dataset: The first layer was trained on dataset containing 30% are of patients affected with bacterial pneumonia, 30% are of patients affected with viral pneumonia and 40% the x-ray images are of healthy people. The second layer was trained on a dataset consisting of posteroanterior lung x-rays of patients whose condition had matured to the ARDS stage.
    
		- Results: 98% accuracy
    

	- Journal article: Genigma(MayemNet): Encryption machine which utilizes cryptography and image steganography as its base concepts to cipher secret messages. It has the retrieval rate of 100%.
    
		- The proposed system called GENigma consists of three neural networks, namely the MayhemNet encoder that encrypts the secret message, the MayhemNet decoder that decrypts the encrypted message and the third one is GEN that generates the cover image for performing image steganography. The proposed system has 100% data retrieval rates at the decoding end and is highly customizable.
    
		- The MayhemNet is a textual as well as numerical data encryption method inspired from the concept of multi layer perceptron (MLP) and extreme learning machine. The MayhemNet is a profoundly deep network of neurons randomly connected to each other, where each neuron has an arbitrarily assigned weight (can be of any numeric data types) and has its own unique activation function. The primary objective kept in mind while creating the MayhemNet is to make it nearly impossible for an attacker to decipher the encrypted data, which is easily achieved by
    
		- MayhemNet. In addition to this, the MayhemNet does not require any prior training, thus, significantly reducing the total training time of the complete architecture.
    
		- The activation functions are randomly allotted to each neuron of the network. In addition to this, the connections between the neurons of two layers are also randomly constructed and the weights are also arbitrarily assigned to each neuron. The possibility of an infinite loop due to the non consecutive layer’s connections is restricted using a variable JUMP that has to be set by the user.
    
-     
    
-     
    

<mark style="background: #D2B3FFA6;">Experience with Amazon Sagemaker</mark>

Have used SageMaker to perform data processing, using the Data Wrangler for large amounts of unprocessed data straight from the chromatograms.

-   Data from Amazon S3, Amazon  Redshift
    
-   Cleaned the data and performed feature engineering, found statistical biases using the SageMaker Clarify
    
-   Visualized the data for better understanding
    
-   Checking performance using a small dataset
    
-   Launched jobs on SageMaker autopilot experiments
    

Experience with Model building and monitoring using SageMaker

-   Trained, tuned and have evaluated models using automatic ML CI/CD pipelines
    
-   Reviewed, configured and tracked model performance for deployment.
    

  
