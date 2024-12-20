**<mark style="background: #D2B3FFA6;">Application</mark>** : Ion mobility spectrometry (IMS) is widely used for national security and law enforcement, placed at key transportation checkpoints such as <mark style="background: #ABF7F7A6;">airports and border crossings for the trace detection of drugs and explosives</mark>. Its simplicity, ease of operation, portability, and rapid analytical performance provides a basis for its real-time screening capability.

1. "IMS has emerged as an essential tool for providing spatially targeted molecular information due to its<mark style="background: #ABF7F7A6;"> high sensitivity, wide molecular coverage and chemical specificity</mark>.” [Page 1](zotero://open-pdf/library/items/EQXWBXRB?page=1&annotation=HQIIPKR4) 
2. "One of the major challenges for mapping the complex cellular milieu is the presence of many isomers and isobars present in these samples” [Page 1](zotero://open-pdf/library/items/EQXWBXRB?page=1&annotation=REKPM4KL) 
3. "This challenge is traditionally addressed using orthogonal LC-based analysis, though, common approaches such as chromatography and electrophoresis are not able to be performed at timescales that are compatible with most imaging applications. Ion mobility offers rapid, gas-phase separations that are readily integrated with IMS workflows” [Page 1](zotero://open-pdf/library/items/EQXWBXRB?page=1&annotation=5DPXULPW) 

##### **<mark style="background: #D2B3FFA6;">Connection with Mass Spectrometry</mark>**
1. IMS is a label-free technology that provides ion maps that are easily correlated to tissue histology for a diverse array of biological specimens.[Page 2](zotero://open-pdf/library/items/EQXWBXRB?page=2&annotation=FBK5FVMH) 
2. <mark style="background: #ADCCFFA6;">"The ion separation capability of ion mobility has major advantages when coupled to imaging mass spectrometry to significantly simplify spectral complexity. Moreover, use of ion mobility for discrimination of ions with similar or equal m/z is invaluable for direct tissue analysis where MS instruments cannot distinguish structural isomers using mass resolving power alone</mark>.” [Page 3](zotero://open-pdf/library/items/EQXWBXRB?page=3&annotation=FIWZENHH) 

##### **<mark style="background: #D2B3FFA6;">Benefits of IMS</mark>**
1. "Ion mobility offers effective separation within milliseconds as well as providing additional molecular information.” [Page 2](zotero://open-pdf/library/items/EQXWBXRB?page=2&annotation=MA9ELQ6F) 

##### **<mark style="background: #D2B3FFA6;">Connection with Drift Time</mark>**
1. "<mark style="background: #ADCCFFA6;">Molecules are affected differentially based on their size, charge and mass where factors such as temperature and pressure can have dramatic effects</mark> as defined by the Mason-Schamp equation. Interactions between analytes and the inert gas are defined differently for each ion mobility technique, but <mark style="background: #ADCCFFA6;">in all cases discrepancies between molecular size-to-charge ratio determine mobility and cause molecules to exit from the ion mobility cell at differing times, referred to as the drift or arrival time.</mark>” [Page 2](zotero://open-pdf/library/items/EQXWBXRB?page=2&annotation=S3YQUPL6) 
2. The drift time of the ion is the time elapsed between the gate opening (or closing) and the maximum of ion peak in the ion mobility spectrum.
3. "Similar to retention times in chromatography, these arrival times can often distinguish molecules of similar mass.” [Page 2](zotero://open-pdf/library/items/EQXWBXRB?page=2&annotation=CQIY6VYN) 

##### **<mark style="background: #D2B3FFA6;">Working</mark>**
1. "ion mobility devices, all achieve molecular separation by exposing analytes to opposing forces where a force is applied to analytes in one direction by collisions with an inert gas and in the opposite direction by a voltage gradient” [Page 2](zotero://open-pdf/library/items/EQXWBXRB?page=2&annotation=Q5FTDUX4)
2.  ![[Images/B95IF9HB.png|550]]

##### **<mark style="background: #D2B3FFA6;">Added Data Dimension</mark>**
1. "To fully describe the molecular drivers of biological processes, structural identification is paramount in understanding their mechanistic underpinnings. The extra data dimension afforded by ion mobility-IMS (x,y,z positions, m/z, and mobility), provides a path forward for improving sensitivity, dynamic range, and specificity in a way that is fully compatible with the imaging experiment.” [Page 11](zotero://open-pdf/library/items/EQXWBXRB?page=11&annotation=7G4BIKI2) 
2. "<mark style="background: #ADCCFFA6;">It is important to note that this added data dimension presents significant computational challenges. As instruments become more sensitive and spatial resolution is improved, the number of pixels (e.g. spectra) collected in a single image increases dramatically which can produce terabyte sized datasets</mark>. This challenge is only compounded by the addition of ion mobility, which adds an extra dimension to the data matrix and exponentially increases its resulting footprint. While this has obvious ramifications for data analysis and processing, it also significantly impacts raw data storage, access, and sharing. There are relatively few databanks willing to store such datasets, especially without significant cost to the authors or journal, creating a considerable barrier for data sharing. As such, this can have a negative impact on scientific reproducibility and collaboration. <mark style="background: #ADCCFFA6;">Some ways of dealing with large data sizes include binning or removing noise profiles to reduce the number of data points across the dataset. While effective at minimizing the data footprint, it removes noise and low intensity peaks, making it difficult to accurately calculate critical data features such as absolute S/N, mass resolution, CCS, and limits of detection/quantitation</mark>.” [Page 11](zotero://open-pdf/library/items/EQXWBXRB?page=11&annotation=YMTR2G4M) 


##### **<mark style="background: #D2B3FFA6;">Major Challenges</mark>**
1. [[Data transformation]] and pre-processing
	- Pivoting the data:  
Example of what pivoting does
![[Screenshot 2022-11-18 at 1.32.51 AM.png|400]]

`Syntax:`
```Python
df_new = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum, fill_value=0)
```

>[!note]
> Pivoting the IMS dataset resulted in a [[Multi Indexing Pandas | Hierarchical indexing]] which is advanced/ Hierarchial indexing from pandas. 

2. Dealing with [[Sparse Datasets]]
4. [[ Feature Hashing or Binning]]
5. [[Clustering]] with 4D data
6. Visualizing 4D data such that it is useful and insightful using [[Plotly]]
