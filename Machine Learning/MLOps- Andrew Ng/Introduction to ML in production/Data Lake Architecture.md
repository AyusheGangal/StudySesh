###### <mark style="background: #D2B3FFA6;">Introduction</mark>
- Data Lake is a term that's appeared in this decade to describe an important component of the data analytics pipeline in the world of [Big Data](https://martinfowler.com/articles/bigData/). The idea is to have a single store for all of the raw data that anyone in an organization might need to analyze. 
- Commonly people use Hadoop to work on the data in the lake, but the concept is broader than just Hadoop.
- <mark style="background: #ABF7F7A6;">Difference between data lake & data warehouse</mark> 
	- The data lake stores _raw_ data, in whatever form the data source provides. There is no assumptions about the schema of the data, each data source can use whatever schema it likes. It's up to the consumers of that data to make sense of that data for their own purposes.
	- This is an important step, many data warehouse initiatives didn't get very far because of schema problems. Data warehouses tend to go with the notion of a single schema for all analytics needs, which can be impractical for anything but the smallest organizations.
	- 
