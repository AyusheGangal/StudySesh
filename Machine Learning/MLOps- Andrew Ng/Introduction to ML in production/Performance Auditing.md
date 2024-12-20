#### <mark style="background: #D2B3FFA6;">Auditing Framework</mark>
1. Brainstorm the ways the system might go wrong
	- Performance on subsets of data (eg, ethnicity, gender)
	- How common are certain errors (eg, FP, FN)
	- Performance on rare classes
	
2. Establish metrics to assess performance against these issues on appropriate slices of data.

Example: Speech Recognition System
1. Brainstorm the ways the system might go wrong: it is very problem dependent
	- Accuracy on different genders and ethnicities' accents.
	- Accuracy on different devices (mics).
	- Prevalence of rude mis-transcriptions. (eg, GAN mis-transcribed as GUN, GANG)
	
2. Establish metrics to assess performance against these issues on appropriate slices of data.
	- Mean accuracy for different genders and major accents.
	- Mean accuracy on different devices.
	- Check for prevalence of offensive words in the output.