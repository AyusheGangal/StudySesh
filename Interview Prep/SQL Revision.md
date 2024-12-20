https://www.interviewbit.com/sql-interview-questions/

## SQL Interview Questions

### 1. What is Database?
A database is an organized collection of data, stored and retrieved digitally from a remote or local computer system. Databases can be vast and complex, and such databases are developed using fixed design and modeling approaches.

### 2. What is DBMS?
DBMS stands for Database Management System. DBMS is a system software responsible for the creation, retrieval, updation, and management of the database. It ensures that our data is consistent, organized, and is easily accessible by serving as an interface between the database and its end-users or application software.

### 3. What is RDBMS? How is it different from DBMS?
RDBMS stands for Relational Database Management System. The key difference [here](https://www.interviewbit.com/blog/difference-between-dbms-and-rdbms/), compared to DBMS, is that RDBMS stores data in the form of a collection of tables, and relations can be defined between the common fields of these tables. Most modern database management systems like MySQL, Microsoft SQL Server, Oracle, IBM DB2, and Amazon Redshift are based on RDBMS.

### 4. what is the difference between SQL and MYSQL
SQL (Structured Query Language) and MySQL are related but distinct concepts. SQL is a standardized programming language used to manage and manipulate relational databases, while MySQL is a specific implementation of a relational database management system (RDBMS) that uses SQL as its primary language.
Here are some key differences between SQL and MySQL:

1.  SQL is a language, while MySQL is a database management system that uses SQL as its primary language.
    
2.  SQL is a standardized language, meaning it is the same across different database management systems. MySQL, on the other hand, is a specific implementation of an RDBMS that uses SQL.
    
3.  MySQL is an open source RDBMS, while SQL is a language that is used with different RDBMS platforms, including open source and proprietary systems.
    
4.  MySQL is known for its ease of use, scalability, and high performance, while SQL is known for its ability to manage and manipulate data in a relational database.
    
5.  MySQL has a variety of features that are specific to its implementation, such as its storage engines and its own extensions to SQL.
    

In summary, SQL is a standardized language used for managing and manipulating relational databases, while MySQL is a specific implementation of an RDBMS that uses SQL as its primary language.

### 5. What are constraints in SQL?
Constraints in SQL are rules that are defined and enforced on columns or tables to maintain the integrity, accuracy, and consistency of data in a relational database.
Here are some common types of constraints in SQL:

1.  NOT NULL: This constraint ensures that a column cannot have a null value.
    
2.  UNIQUE: This constraint ensures that each value in a column is unique, and no duplicates are allowed.
    
3.  PRIMARY KEY: This constraint is a combination of the NOT NULL and UNIQUE constraints and ensures that a column or set of columns uniquely identifies each row in a table.
    
4.  FOREIGN KEY: This constraint creates a relationship between two tables, where the values in one table's column(s) must match the values in another table's column(s).
    
5.  CHECK: This constraint allows the definition of a condition that each row must satisfy to be inserted or updated in a table.
    
6.  DEFAULT: This constraint sets a default value for a column if no value is specified during an INSERT operation.
    

Constraints help ensure that the data in a relational database is consistent, accurate, and meets the requirements of the application. They also help prevent errors and maintain data integrity.

### 6. How does indexing work in SQL? what are they
Indexing in SQL is a technique used to improve the performance of queries on large databases. An index is a data structure that is created on one or more columns of a table, which allows for faster retrieval of data.

When an index is created on a column, a separate data structure is created that contains a copy of the column data along with a pointer to the actual data row in the table. When a query is executed that includes the indexed column(s), the database engine can use the index to quickly locate the relevant rows in the table, rather than scanning the entire table.

Here are some types of indexes in SQL:
1.  Clustered index: This is an index that determines the physical order of the data in the table. Each table can only have one clustered index, which is typically created on the primary key column.
2.  Non-clustered index: This is an index that is created on a non-primary key column. It contains a copy of the indexed column data along with a pointer to the actual data row in the table.
3.  Unique index: This is an index that enforces uniqueness on the column(s) it is created on.
4.  Composite index: This is an index that is created on multiple columns in a table.

Indexes can significantly improve the performance of SQL queries, but they also require additional storage space and can slow down INSERT, UPDATE, and DELETE operations. It is important to carefully consider the columns to be indexed and the type of index to use for a particular table to achieve the best performance.

### 7. What is the difference between DDL and DML commands
DDL (Data Definition Language) and DML (Data Manipulation Language) are two types of SQL commands that serve different purposes in a relational database.

DDL commands are used to define the structure and schema of a database, including creating, modifying, and deleting database objects such as tables, views, indexes, and constraints. Some common DDL commands include CREATE, ALTER, and DROP.

On the other hand, DML commands are used to manipulate the data stored in a database. This includes inserting, updating, and deleting rows in tables, as well as querying data from tables. Some common DML commands include SELECT, INSERT, UPDATE, and DELETE.

Here are some key differences between DDL and DML commands:
1.  Purpose: DDL commands are used to define the structure and schema of a database, while DML commands are used to manipulate the data stored in a database.
2.  Syntax: DDL commands use syntax that is specific to the creation and modification of database objects, while DML commands use syntax that is specific to manipulating data in tables.
3.  Impact: DDL commands can have a significant impact on the database schema and structure, and can result in the loss of data if not used correctly. DML commands can also impact data integrity, but the impact is limited to the rows being manipulated.    

In summary, DDL commands are used to define the structure and schema of a database, while DML commands are used to manipulate the data stored in a database.

### 8. What is the difference between ALTER, MODIFY and UPDATE?

ALTER, MODIFY, and UPDATE are SQL commands that serve different purposes in manipulating database objects and data. Here are the differences between the three commands:

1.  ALTER: The ALTER command is a DDL (Data Definition Language) command used to modify the structure of an existing database object, such as a table, view, index, or schema. The ALTER command can be used to perform a wide range of modifications, such as adding, modifying, or deleting columns in a table, changing the data type of a column, adding or dropping constraints, renaming a database object, modifying the storage options of a table or index, and modifying the schema of a database object.
    
2.  MODIFY: The MODIFY command is typically used in the context of the ALTER command, and is also a DDL command used to modify the structure of an existing database object. In most SQL implementations, the MODIFY command is used to alter the properties of a column in a table, such as its data type, length, nullability, or default value.
    
3.  UPDATE: The UPDATE command is a DML (Data Manipulation Language) command used to modify the data stored in a table. The UPDATE command is used to change the values of one or more columns in one or more rows of a table. The UPDATE command uses a WHERE clause to specify which rows to update based on a condition.
    

In summary, ALTER is used to modify the structure of a database object, MODIFY is typically used in the context of ALTER to modify the properties of a column in a table, and UPDATE is used to modify the data stored in a table.

### 9. can you give examples for all three?

Certainly! Here are some examples for each of the three SQL commands:

1.  `ALTER`:
    Example 1: Add a new column to an existing table: Used with keyword `ADD`
    ```sql
ALTER TABLE my_table ADD COLUMN new_column INTEGER;
```
    
    Example 2: Modify the data type of an existing column in a table: Used with keyword `ALTER`
    ```sql
ALTER TABLE my_table ALTER COLUMN existing_column TEXT;
```

2.  `MODIFY`:
    Example 1: Modify the data type of an existing column in a table: Used with keyword `ALTER`
    ```sql
ALTER TABLE my_table MODIFY COLUMN existing_column TEXT;
```
    
    Example 2: Modify the default value of an existing column in a table: Used with keyword `SET` and `MODIFY`
    ```sql
ALTER TABLE my_table MODIFY COLUMN existing_column SET DEFAULT 'new_default_value';
```
    
3.  UPDATE:
    Example 1: Update the value of a column in all rows of a table: Used with keyword `SET`
    ```sql
UPDATE my_table SET column_to_update = 'new_value';
```
    
    Example 2: Update the value of a column in a specific row of a table: Used with keyword `SET` and `WHERE`
    ```sql
UPDATE my_table SET column_to_update = 'new_value' WHERE id = 123;
```


### 10. What is a Primary Key?

The PRIMARY KEY constraint uniquely identifies each row in a table. It must contain UNIQUE values and has an implicit NOT NULL constraint.  
A table in SQL is strictly restricted to have one and only one primary key, which is comprised of single or multiple fields (columns).

```sql
CREATE TABLE Students (   /* Create table with a single field as primary key */
   ID INT NOT NULL
   Name VARCHAR(255)
   PRIMARY KEY (ID)
);

CREATE TABLE Students (   /* Create table with multiple fields as primary key */
   ID INT NOT NULL
   LastName VARCHAR(255)
   FirstName VARCHAR(255) NOT NULL,
   CONSTRAINT PK_Student
   PRIMARY KEY (ID, FirstName)
);

ALTER TABLE Students   /* Set a column as primary key */
ADD PRIMARY KEY (ID);
ALTER TABLE Students   /* Set multiple columns as primary key */
ADD CONSTRAINT PK_Student   /*Naming a Primary Key*/
PRIMARY KEY (ID, FirstName);
```

In a relational database management system (RDBMS), a table can only have one primary key. The primary key is a unique identifier for each row in the table, and it must be unique and non-null.

While a table cannot have multiple primary keys, it can have a composite primary key, which is a primary key that consists of two or more columns in the table. A composite primary key is created by specifying multiple columns in the primary key constraint declaration, like this:

```sql
CREATE TABLE my_table (column1 INT, 
					   column2 VARCHAR(50), 
					   column3 DATE, 
					   PRIMARY KEY (column1, column2));
````

In this example, the primary key for the `my_table` table consists of the `column1` and `column2` columns. This means that the combination of values in these two columns must be unique for each row in the table.

While composite primary keys can provide more flexibility than a single-column primary key, they can also make it more difficult to query the table, especially if the primary key consists of several columns. Therefore, it's important to carefully consider the design of the primary key when creating a new table in an RDBMS.


### 11. What is a UNIQUE constraint?
A UNIQUE constraint ensures that all values in a column are different. This provides uniqueness for the column(s) and helps identify each row uniquely. Unlike primary key, there can be multiple unique constraints defined per table. The code syntax for UNIQUE is quite similar to that of PRIMARY KEY and can be used interchangeably.

```sql
CREATE TABLE Students (   /* Create table with a single field as unique */
   ID INT NOT NULL UNIQUE
   Name VARCHAR(255)
);

CREATE TABLE Students (   /* Create table with multiple fields as unique */
   ID INT NOT NULL
   LastName VARCHAR(255)
   FirstName VARCHAR(255) NOT NULL
   CONSTRAINT PK_Student
   UNIQUE (ID, FirstName)
);

ALTER TABLE Students   /* Set a column as unique */
ADD UNIQUE (ID);
ALTER TABLE Students   /* Set multiple columns as unique */
ADD CONSTRAINT PK_Student   /* Naming a unique constraint */
UNIQUE (ID, FirstName);
```


### 12. What is a Foreign Key?
A FOREIGN KEY comprises of single or collection of fields in a table that essentially refers to the PRIMARY KEY in another table. Foreign key constraint ensures referential integrity in the relation between two tables.  
The table with the foreign key constraint is labeled as the child table, and the table containing the candidate key is labeled as the referenced or parent table.

```sql
CREATE TABLE Students (   /* Create table with foreign key - Way 1 */
   ID INT NOT NULL
   Name VARCHAR(255)
   LibraryID INT
   PRIMARY KEY (ID)
   FOREIGN KEY (Library_ID) REFERENCES Library(LibraryID)
);

CREATE TABLE Students (   /* Create table with foreign key - Way 2 */
   ID INT NOT NULL PRIMARY KEY
   Name VARCHAR(255)
   LibraryID INT FOREIGN KEY (Library_ID) REFERENCES Library(LibraryID)
);

ALTER TABLE Students   /* Add a new foreign key */
ADD FOREIGN KEY (LibraryID)
REFERENCES Library (LibraryID);
```


###  13. What is a Join? List its different types.
The [**SQL Join**](https://www.scaler.com/topics/joins-in-sql/) clause is used to combine records (rows) from two or more tables in a SQL database based on a related column between the two.

There are four different types of JOINs in SQL:
-   **(INNER) JOIN:** Retrieves records that have matching values in both tables involved in the join. This is the widely used join for queries.
```sql
SELECT *
FROM Table_A
JOIN Table_B;
SELECT *
FROM Table_A
INNER JOIN Table_B;
```

-   **LEFT (OUTER) JOIN:** Retrieves all the records/rows from the left and the matched records/rows from the right table.
```sql
SELECT *
FROM Table_A A
LEFT JOIN Table_B B
ON A.col = B.col;
```

-   **RIGHT (OUTER) JOIN:** Retrieves all the records/rows from the right and the matched records/rows from the left table.
```sql
SELECT *
FROM Table_A A
RIGHT JOIN Table_B B
ON A.col = B.col;
```

-   **FULL (OUTER) JOIN:** Retrieves all the records where there is a match in either the left or right table.
```sql
SELECT *
FROM Table_A A
FULL JOIN Table_B B
ON A.col = B.col;
```

