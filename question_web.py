import streamlit as st
import random

# --- 1. Data Fetching ---
def fetch_dynamic_questions():
    # All your questions, options, and correct answers are stored here
    massive_question_bank = [
        {
            "question": "What is the output of `type(5 / 2)` in Python 3?",
            "options": ["A) int", "B) float", "C) double", "D) decimal"],
            "correct_answer": "B"
        },
        {
            "question": "What will the following code output?\n`print(\"Exeter\"[-2:])`",
            "options": ["A) er", "B) te", "C) re", "D) Error"],
            "correct_answer": "A"
        },
        {
            "question": "Which pandas method is used to remove missing values (NaN) from a DataFrame?",
            "options": ["A) drop_nulls()", "B) remove_na()", "C) dropna()", "D) fillna()"],
            "correct_answer": "C"
        },
        {
            "question": "What is the output of the expression `3 * 'AB'` in Python?",
            "options": ["A) 'ABABAB'", "B) 'AB3'", "C) Error", "D) '3AB'"],
            "correct_answer": "A"
        },
        {
            "question": "Which method is used to remove whitespace from the beginning and end of a string?",
            "options": ["A) .lstrip()", "B) .strip()", "C) .trim()", "D) .replace()"],
            "correct_answer": "B"
        },
        {
            "question": "What is the correct syntax for a list comprehension that returns the squares of numbers from 0 to 4?",
            "options": ["A) [x**2 for x in range(5)]", "B) [x^2 for x in range(4)]", "C) [x**2 in range(5)]", "D) [x*2 for x in range(5)]"],
            "correct_answer": "A"
        },
        {
            "question": "When accessing a dictionary, what is the advantage of using `my_dict.get('key')` over `my_dict['key']`?",
            "options": ["A) It is significantly faster.", "B) It returns None instead of a KeyError if the key does not exist.", "C) It allows you to modify the key.", "D) It automatically sorts the dictionary."],
            "correct_answer": "B"
        },
        {
            "question": "In a Python function definition, what does `*args` allow you to do?",
            "options": ["A) Pass keyword arguments.", "B) Unpack a dictionary.", "C) Pass a variable number of positional arguments.", "D) Force the function to return a tuple."],
            "correct_answer": "C"
        },
        {
            "question": "Which statement is used to immediately exit a `while` loop, skipping the rest of the code block?",
            "options": ["A) exit", "B) pass", "C) continue", "D) break"],
            "correct_answer": "D"
        },
        {
            "question": "You need to load a dataset named `sales_data.csv` into a pandas DataFrame. Which function do you use?",
            "options": ["A) pd.read_file('sales_data.csv')", "B) pd.load_csv('sales_data.csv')", "C) pd.read_csv('sales_data.csv')", "D) pd.open('sales_data.csv')"],
            "correct_answer": "C"
        },
        {
            "question": "You are analyzing student engagement data. How do you filter a DataFrame `df` to show only students with an 'Attendance' score exactly equal to 100?",
            "options": ["A) df[df['Attendance'] = 100]", "B) df[df['Attendance'] == 100]", "C) df.filter('Attendance' == 100)", "D) df['Attendance'].equals(100)"],
            "correct_answer": "B"
        },
        {
            "question": "Which pandas method allows you to fill all missing values (NaN) with the number 0?",
            "options": ["A) df.replace_na(0)", "B) df.fill_nulls(0)", "C) df.dropna(0)", "D) df.fillna(0)"],
            "correct_answer": "D"
        },
        {
            "question": "To find the average 'Revenue' per 'Region', which pandas operation should you use?",
            "options": ["A) df.mean('Region')['Revenue']", "B) df.groupby('Region')['Revenue'].mean()", "C) df['Revenue'].groupby('Region').avg()", "D) df.aggregate('Region', 'Revenue', 'mean')"],
            "correct_answer": "B"
        },
        {
            "question": "How do you sort a DataFrame `df` by the 'Profit' column in descending order?",
            "options": ["A) df.sort('Profit', descending=True)", "B) df.order_by('Profit', asc=False)", "C) df.sort_values(by='Profit', ascending=False)", "D) df.sort_values('Profit', descending=True)"],
            "correct_answer": "C"
        },
        {
            "question": "What does the `df.shape` attribute return?",
            "options": ["A) A tuple representing the memory address of the DataFrame.", "B) The total number of non-null cells in the DataFrame.", "C) A tuple containing the number of rows and columns (rows, columns).", "D) A list of all column names."],
            "correct_answer": "C"
        },
        {
            "question": "Which pandas method is used to apply a custom Python function along an axis of the DataFrame (e.g., to every row or column)?",
            "options": ["A) .map()", "B) .apply()", "C) .execute()", "D) .run()"],
            "correct_answer": "B"
        },
        {
            "question": "When using `pd.merge()`, how do you specify a Left Join?",
            "options": ["A) join='left'", "B) type='left'", "C) how='left'", "D) method='left'"],
            "correct_answer": "C"
        },
        {
            "question": "What is the NumPy function used to create an array of evenly spaced values within a given interval, similar to Python's built-in `range()`?",
            "options": ["A) np.linspace()", "B) np.interval()", "C) np.arange()", "D) np.range()"],
            "correct_answer": "C"
        },
        {
            "question": "How do you create a 3x3 NumPy array filled entirely with zeros?",
            "options": ["A) np.zeros(3, 3)", "B) np.zeros((3, 3))", "C) np.empty(3, 3)", "D) np.array(0, (3, 3))"],
            "correct_answer": "B"
        },
        {
            "question": "What is 'broadcasting' in NumPy?",
            "options": ["A) Sending data from NumPy to pandas.", "B) The ability to apply operations on arrays of different shapes.", "C) Converting a list to an array.", "D) Printing an array to the console."],
            "correct_answer": "B"
        },
        {
            "question": "You are building a learning analytics dashboard and want to plot 'Study Hours' vs 'Exam Scores' as individual data points. Which matplotlib function is best?",
            "options": ["A) plt.plot()", "B) plt.scatter()", "C) plt.bar()", "D) plt.points()"],
            "correct_answer": "B"
        },
        {
            "question": "How do you set the main title of a matplotlib plot?",
            "options": ["A) plt.heading('Title')", "B) plt.set_title('Title')", "C) plt.title('Title')", "D) plt.header('Title')"],
            "correct_answer": "C"
        },
        {
            "question": "In seaborn, which function is used to show point estimates and confidence intervals as rectangular bars?",
            "options": ["A) sns.histplot()", "B) sns.boxplot()", "C) sns.barplot()", "D) sns.lineplot()"],
            "correct_answer": "C"
        },
        {
            "question": "Which of the following creates a boolean value of `False`?",
            "options": ["A) bool(1)", "B) bool(\"False\")", "C) bool([])", "D) bool([0])"],
            "correct_answer": "C"
        },
        {
            "question": "What does the `df.describe()` method do in pandas?",
            "options": ["A) Generates descriptive statistics like count, mean, min, and max for numerical columns.", "B) Prints the column names and data types.", "C) Returns a textual summary of the DataFrame's purpose.", "D) Describes the missing values in the dataset."],
            "correct_answer": "A"
        },
        {
            "question": "Which Python block lets you test a block of code for errors?",
            "options": ["A) try", "B) except", "C) test", "D) catch"],
            "correct_answer": "A"
        },
        {
            "question": "How do you save a matplotlib figure to your local directory as an image file?",
            "options": ["A) plt.export('figure.png')", "B) plt.save('figure.png')", "C) plt.savefig('figure.png')", "D) plt.write_image('figure.png')"],
            "correct_answer": "C"
        },
        {
            "question": "If `x = [1, 2, 3]`, what is `x * 2`?",
            "options": ["A) [2, 4, 6]", "B) [1, 2, 3, 1, 2, 3]", "C) [[1, 2, 3], [1, 2, 3]]", "D) Error"],
            "correct_answer": "B"
        },
        # --- BATCH 1: Advanced Pandas, Data Cleaning & Business Logic ---
        {
            "question": "You have a DataFrame `sales` with daily data. How do you resample this data to find the total monthly revenue?",
            "options": ["A) sales.resample('M').sum()", "B) sales.groupby('Month').sum()", "C) sales.resample('Monthly').add()", "D) sales.aggregate('M', sum)"],
            "correct_answer": "A"
        },
        {
            "question": "In pandas, what is the purpose of the `pd.to_datetime()` function when cleaning a customer database?",
            "options": ["A) To calculate the time taken to run a script.", "B) To convert string representations of dates into workable datetime objects.", "C) To automatically set the DataFrame index to the current system time.", "D) To sort the DataFrame chronologically."],
            "correct_answer": "B"
        },
        {
            "question": "You are analyzing customer churn. Which pandas method replaces all NaN values in the 'Churn_Status' column with 'Active'?",
            "options": ["A) df['Churn_Status'].replace_na('Active', inplace=True)", "B) df['Churn_Status'].fillna('Active', inplace=True)", "C) df.dropna(column='Churn_Status', fill='Active')", "D) df['Churn_Status'].insert('Active')"],
            "correct_answer": "B"
        },
        {
            "question": "What does `df.groupby(['Region', 'Product_Category'])['Profit'].mean()` do?",
            "options": ["A) Averages the profit across all regions ignoring categories.", "B) Calculates the total profit for each region and category.", "C) Calculates the average profit for every unique combination of Region and Product Category.", "D) Sorts the DataFrame by Region, then Category, then Profit."],
            "correct_answer": "C"
        },
        {
            "question": "In a retail dataset, how do you find the number of unique customers who made a purchase?",
            "options": ["A) df['Customer_ID'].count()", "B) df['Customer_ID'].unique_count()", "C) df['Customer_ID'].nunique()", "D) df.count_distinct('Customer_ID')"],
            "correct_answer": "C"
        },
        {
            "question": "Which scikit-learn function is commonly used to split a business dataset into training and testing sets for a predictive model?",
            "options": ["A) model_selection.split_data", "B) train_test_split()", "C) sklearn.divide_data()", "D) cross_val_split()"],
            "correct_answer": "B"
        },
        {
            "question": "You want to merge an 'Employees' DataFrame with a 'Salaries' DataFrame, keeping all employees even if their salary data is missing. Which join type do you use?",
            "options": ["A) how='inner'", "B) how='outer'", "C) how='left'", "D) how='right'"],
            "correct_answer": "C"
        },
        {
            "question": "What is the result of `df.corr()` in pandas?",
            "options": ["A) Corrects syntax errors in column names.", "B) Computes pairwise correlation of columns, excluding NA/null values.", "C) Returns the corresponding index of the maximum value.", "D) Calculates the variance of the numerical columns."],
            "correct_answer": "B"
        },
        {
            "question": "When visualizing quarterly revenue trends, which matplotlib function creates a figure with a specific size (e.g., 10x6 inches)?",
            "options": ["A) plt.figure(size=(10, 6))", "B) plt.plot(figsize=(10, 6))", "C) plt.figure(figsize=(10, 6))", "D) plt.window(10, 6)"],
            "correct_answer": "C"
        },
        {
            "question": "You have an array of daily stock returns. Which NumPy function calculates the cumulative product to find the overall return over time?",
            "options": ["A) np.cumprod()", "B) np.product()", "C) np.cumsum()", "D) np.multiply_all()"],
            "correct_answer": "A"
        },
        {
            "question": "How do you drop a column named 'Tax_ID' from a DataFrame `df` permanently?",
            "options": ["A) df.drop('Tax_ID', axis=1, inplace=True)", "B) df.remove('Tax_ID')", "C) df.delete_column('Tax_ID')", "D) df.drop('Tax_ID', axis=0)"],
            "correct_answer": "A"
        },
        {
            "question": "What is the primary difference between `append()` and `extend()` when working with Python lists holding business KPI data?",
            "options": ["A) `append` adds multiple elements, `extend` adds one.", "B) `append` adds its argument as a single element to the end of a list, `extend` iterates over its argument and adds each element to the list.", "C) There is no difference.", "D) `extend` works on dictionaries, `append` works on lists."],
            "correct_answer": "B"
        },
        {
            "question": "If you apply `.value_counts(normalize=True)` to a 'Customer_Segment' column, what does it return?",
            "options": ["A) The total count of each segment.", "B) The relative frequencies (proportions) of each segment from 0 to 1.", "C) The alphabetical list of segments.", "D) Normalizes the text (lowercase and strips spaces)."],
            "correct_answer": "B"
        },
        {
            "question": "In Seaborn, what does a heatmap typically require as input to visualize correlations between multiple business metrics?",
            "options": ["A) A 1D array or list.", "B) A string of column names.", "C) A 2D rectangular dataset, such as a pandas correlation matrix.", "D) A SQL database connection."],
            "correct_answer": "C"
        },
        {
            "question": "You are writing a pricing algorithm. What does `15 % 4` evaluate to in Python?",
            "options": ["A) 3.75", "B) 3", "C) 15", "D) 4"],
            "correct_answer": "B"
        },
        {
            "question": "Which pandas property gives you the names of the columns in a DataFrame?",
            "options": ["A) df.headers", "B) df.keys()", "C) df.columns", "D) df.names"],
            "correct_answer": "C"
        },
        {
            "question": "To convert a column of string prices (e.g., '$1,000') to numerical floats, what must you typically do first in pandas?",
            "options": ["A) Use `astype(float)` directly.", "B) Remove the '$' and ',' characters using `.str.replace()`.", "C) Use `pd.to_numeric()` directly.", "D) Divide the string by 1."],
            "correct_answer": "B"
        },
        {
            "question": "What is the default standard deviation calculation behavior difference between NumPy (`np.std()`) and pandas (`df.std()`)?",
            "options": ["A) NumPy uses sample std (ddof=1), pandas uses population std (ddof=0).", "B) NumPy uses population std (ddof=0), pandas uses sample std (ddof=1).", "C) NumPy cannot calculate standard deviation.", "D) There is no difference."],
            "correct_answer": "B"
        },
        {
            "question": "How do you rename a column from 'Old_Price' to 'New_Price' in a DataFrame `df`?",
            "options": ["A) df.columns['Old_Price'] = 'New_Price'", "B) df.rename(columns={'Old_Price': 'New_Price'}, inplace=True)", "C) df.set_column('Old_Price', 'New_Price')", "D) df.change_name('Old_Price', 'New_Price')"],
            "correct_answer": "B"
        },
        {
            "question": "In a logistics dataset, you want to identify orders that are either 'Late' OR cost more than £500. Which logical operator do you use in pandas?",
            "options": ["A) and", "B) &", "C) or", "D) |"],
            "correct_answer": "D"
        },
        {
            "question": "When defining a function to calculate Compound Annual Growth Rate (CAGR), what does the `return` statement do?",
            "options": ["A) Prints the result to the console.", "B) Exits the function and passes the result back to the caller.", "C) Restarts the function loop.", "D) Saves the variable to the global namespace."],
            "correct_answer": "B"
        },
        {
            "question": "Which seaborn plot is best for visualizing the distribution of employee salaries across different departments?",
            "options": ["A) sns.lineplot()", "B) sns.boxplot()", "C) sns.scatterplot()", "D) sns.pieplot()"],
            "correct_answer": "B"
        },
        {
            "question": "You have a DataFrame `inventory`. How do you select all rows where the 'Stock' is less than 10?",
            "options": ["A) inventory.filter('Stock' < 10)", "B) inventory[inventory['Stock'] < 10]", "C) inventory.query('Stock' < 10)", "D) Both B and C are correct."],
            "correct_answer": "D"
        },
        {
            "question": "What is the purpose of the `random_state` parameter in machine learning functions like `train_test_split` or `RandomForestRegressor`?",
            "options": ["A) To introduce maximum randomness for better accuracy.", "B) To ensure the results are reproducible across different runs.", "C) To define the percentage of data used for testing.", "D) To randomize the column order."],
            "correct_answer": "B"
        },
        {
            "question": "How do you extract the year from a datetime column 'Order_Date' in pandas?",
            "options": ["A) df['Order_Date'].dt.year", "B) df['Order_Date'].year()", "C) df['Order_Date'].get_year()", "D) year(df['Order_Date'])"],
            "correct_answer": "A"
        },
        {
            "question": "Which pandas function is ideal for creating a pivot table to summarize total sales by region and product?",
            "options": ["A) pd.cross_tab()", "B) df.pivot_table()", "C) df.unstack()", "D) df.melt()"],
            "correct_answer": "B"
        },
        {
            "question": "If you have a list of monthly marketing spends, how do you slice the list to get only the last three months?",
            "options": ["A) spends[3:]", "B) spends[:-3]", "C) spends[-3:]", "D) spends[0:3]"],
            "correct_answer": "C"
        },
        {
            "question": "What does a `KeyError` typically mean when working with a pandas DataFrame?",
            "options": ["A) You forgot a closing parenthesis.", "B) You are trying to access a column name or dictionary key that does not exist.", "C) The data type is incorrect for the mathematical operation.", "D) The API key for data extraction is invalid."],
            "correct_answer": "B"
        },
        {
            "question": "Which NumPy function is used to compute the dot product of two matrices, a common operation in advanced predictive analytics?",
            "options": ["A) np.cross()", "B) np.multiply()", "C) np.dot()", "D) np.matrix_multiply()"],
            "correct_answer": "C"
        },
        {
            "question": "You want to find the row with the absolute highest revenue in a DataFrame. Which method returns the index (row label) of that maximum value?",
            "options": ["A) df['Revenue'].max()", "B) df['Revenue'].idxmax()", "C) df['Revenue'].argmax()", "D) df['Revenue'].top()"],
            "correct_answer": "B"
        },
        # --- BATCH 2: Predictive Modeling, Advanced Pandas & Scikit-learn ---
        {
            "question": "In scikit-learn, what is the standard method used to train a machine learning model, such as a `LinearRegression` algorithm, on your training data?",
            "options": ["A) model.train(X, y)", "B) model.fit(X, y)", "C) model.predict(X, y)", "D) model.build(X, y)"],
            "correct_answer": "B"
        },
        {
            "question": "When preparing categorical data (like 'Department' names) for a machine learning model, which pandas function creates binary columns for each category (One-Hot Encoding)?",
            "options": ["A) pd.categorize()", "B) pd.to_numeric()", "C) pd.get_dummies()", "D) pd.binary_encode()"],
            "correct_answer": "C"
        },
        {
            "question": "What is the primary purpose of `StandardScaler()` in a data preprocessing pipeline?",
            "options": ["A) To convert all strings to floats.", "B) To normalize features by removing the mean and scaling to unit variance.", "C) To scale all values between 0 and 1 exactly.", "D) To automatically drop outliers from the dataset."],
            "correct_answer": "B"
        },
        {
            "question": "Which scikit-learn metric is best used to evaluate the performance of a continuous predictive model, such as predicting next month's sales revenue?",
            "options": ["A) accuracy_score", "B) f1_score", "C) confusion_matrix", "D) mean_squared_error"],
            "correct_answer": "D"
        },
        {
            "question": "You are building a churn prediction model (Binary Classification). Which algorithm is most appropriate for predicting if a customer will leave (1) or stay (0)?",
            "options": ["A) LinearRegression", "B) LogisticRegression", "C) KMeans", "D) PCA"],
            "correct_answer": "B"
        },
        {
            "question": "In a Python `for` loop, what does the `zip()` function allow you to do?",
            "options": ["A) Compress a large pandas DataFrame to save memory.", "B) Iterate over two or more lists simultaneously, pairing their elements.", "C) Combine multiple files into a zip archive.", "D) Sort a list in descending order instantly."],
            "correct_answer": "B"
        },
        {
            "question": "You have a list of prices: `[10, 20, 30]`. You want to apply a 10% tax to each using a list comprehension. Which syntax is correct?",
            "options": ["A) [p * 1.10 for p in prices]", "B) [prices * 1.10]", "C) [for p in prices: p * 1.10]", "D) list(prices * 1.10)"],
            "correct_answer": "A"
        },
        {
            "question": "What does the `enumerate()` function return when used in a loop?",
            "options": ["A) The sum of the list elements.", "B) Both the index (counter) and the value of the item in the iterable.", "C) Only the items that are numbers.", "D) A randomized order of the list elements."],
            "correct_answer": "B"
        },
        {
            "question": "How do you apply a custom lambda function to calculate profit margins across an entire 'Revenue' column in a DataFrame?",
            "options": ["A) df['Revenue'].apply(lambda x: x * 0.25)", "B) df['Revenue'].map(lambda: x * 0.25)", "C) apply(df['Revenue'], lambda x: x * 0.25)", "D) df['Revenue'] = lambda x: x * 0.25"],
            "correct_answer": "A"
        },
        {
            "question": "To analyze month-over-month sales growth, you need to compare the current row to the previous row. Which pandas method shifts the data down by one period?",
            "options": ["A) df.move()", "B) df.lag()", "C) df.shift(1)", "D) df.next(-1)"],
            "correct_answer": "C"
        },
        {
            "question": "When evaluating a classification model, what does the 'Recall' metric measure?",
            "options": ["A) The ratio of correctly predicted positive observations to the total actual positives.", "B) The ratio of correctly predicted positive observations to the total predicted positives.", "C) The total number of true negatives.", "D) The speed at which the model makes a prediction."],
            "correct_answer": "A"
        },
        {
            "question": "If `y_true = [1, 0, 1]` and `y_pred = [1, 1, 1]`, what is the accuracy of the model?",
            "options": ["A) 100%", "B) 33.3%", "C) 66.6%", "D) 0%"],
            "correct_answer": "C"
        },
        {
            "question": "In pandas, what is the purpose of `df.rolling(window=7).mean()`?",
            "options": ["A) Calculates the mean of every 7th row.", "B) Calculates a 7-day (or 7-period) moving average.", "C) Rounds all numbers in the DataFrame to 7 decimal places.", "D) Groups the data by week and calculates the mean."],
            "correct_answer": "B"
        },
        {
            "question": "Which function allows you to combine two DataFrames vertically (stacking one on top of the other)?",
            "options": ["A) pd.concat([df1, df2], axis=0)", "B) pd.merge(df1, df2, how='vertical')", "C) pd.concat([df1, df2], axis=1)", "D) df1.join(df2)"],
            "correct_answer": "A"
        },
        {
            "question": "In a Jupyter Notebook, what is the output of `type(np.nan)`?",
            "options": ["A) NoneType", "B) float", "C) string", "D) int"],
            "correct_answer": "B"
        },
        {
            "question": "You are creating a dashboard and need to display two matplotlib plots side-by-side. Which function sets up this grid?",
            "options": ["A) plt.grid(1, 2)", "B) plt.subplots(1, 2)", "C) plt.figure(columns=2)", "D) plt.twinx()"],
            "correct_answer": "B"
        },
        {
            "question": "What is the primary use case for K-Means clustering in business analytics?",
            "options": ["A) Predicting future stock prices.", "B) Classifying images into categories.", "C) Grouping customers into distinct segments based on purchasing behavior.", "D) Forecasting monthly sales volume."],
            "correct_answer": "C"
        },
        {
            "question": "When performing K-Means clustering, what is the 'Elbow Method' used for?",
            "options": ["A) Determining the optimal number of clusters (k).", "B) Removing outliers from the dataset.", "C) Scaling the features to ensure equal weighting.", "D) Calculating the distance between centroids."],
            "correct_answer": "A"
        },
        {
            "question": "Which of the following creates a dictionary with keys 'A' and 'B' mapped to values 1 and 2, respectively?",
            "options": ["A) dict('A'=1, 'B'=2)", "B) {'A': 1, 'B': 2}", "C) ['A': 1, 'B': 2]", "D) dict(['A', 1], ['B', 2])"],
            "correct_answer": "B"
        },
        {
            "question": "You want to find the number of days between two dates in pandas. If you subtract a datetime column from another, what is the resulting data type?",
            "options": ["A) int", "B) float", "C) timedelta", "D) datetime"],
            "correct_answer": "C"
        },
        {
            "question": "Which pandas method converts a DataFrame from a wide format (many columns) to a long format (fewer columns, more rows)?",
            "options": ["A) df.pivot()", "B) df.melt()", "C) df.transpose()", "D) df.stack_all()"],
            "correct_answer": "B"
        },
        {
            "question": "In Python, what is a generator?",
            "options": ["A) A function that returns a list of all random numbers.", "B) A function that yields items one at a time using the `yield` keyword, saving memory.", "C) A class used to create new DataFrames.", "D) An algorithm that generates synthetic business data."],
            "correct_answer": "B"
        },
        {
            "question": "What is the purpose of the `random.seed()` or `np.random.seed()` function?",
            "options": ["A) To generate a truly random, unpredictable number.", "B) To initialize the random number generator so that it produces the exact same sequence of numbers each time it is run.", "C) To securely encrypt passwords.", "D) To randomize the order of rows in a DataFrame."],
            "correct_answer": "B"
        },
        {
            "question": "You built a predictive model, but it performs incredibly well on training data and poorly on testing data. What is this phenomenon called?",
            "options": ["A) Underfitting", "B) Bias", "C) Overfitting", "D) Collinearity"],
            "correct_answer": "C"
        },
        {
            "question": "Which regularization technique in linear regression adds a penalty equal to the absolute value of the magnitude of coefficients (L1 penalty)?",
            "options": ["A) Ridge Regression", "B) Elastic Net", "C) Lasso Regression", "D) Logistic Regression"],
            "correct_answer": "C"
        },
        {
            "question": "How do you drop any duplicate rows from a pandas DataFrame `df` based on the 'Transaction_ID' column?",
            "options": ["A) df.drop_duplicates(subset=['Transaction_ID'])", "B) df.unique('Transaction_ID')", "C) df.remove_duplicates('Transaction_ID')", "D) df.distinct('Transaction_ID')"],
            "correct_answer": "A"
        },
        {
            "question": "Which Seaborn visualization plots pairwise relationships across an entire dataset, creating a grid of scatterplots and histograms?",
            "options": ["A) sns.jointplot()", "B) sns.pairplot()", "C) sns.matrixplot()", "D) sns.gridplot()"],
            "correct_answer": "B"
        },
        {
            "question": "In a try/except block, what is the purpose of the `finally` clause?",
            "options": ["A) It executes only if an error occurs.", "B) It executes only if no errors occur.", "C) It executes regardless of whether an exception was raised or not, often used for cleanup tasks.", "D) It forces the program to crash immediately."],
            "correct_answer": "C"
        },
        {
            "question": "If you have a pandas Series `s = pd.Series([10, 20, 30])`, what does `s.apply(lambda x: True if x > 15 else False)` return?",
            "options": ["A) [False, True, True]", "B) A Series of boolean values: False, True, True", "C) [20, 30]", "D) A TypeError"],
            "correct_answer": "B"
        },
        {
            "question": "What is a major advantage of using a Random Forest algorithm over a single Decision Tree in business analytics?",
            "options": ["A) It is much faster to train.", "B) It produces a single, easily interpretable tree graphic.", "C) It reduces the risk of overfitting by averaging multiple trees.", "D) It requires no data preprocessing whatsoever."],
            "correct_answer": "C"
        },
        # --- BATCH 3: APIs, JSON, and Web Data ---
        {
            "question": "When making an API call using the `requests` library to fetch financial data, which HTTP status code indicates a successful request?",
            "options": ["A) 404", "B) 500", "C) 200", "D) 301"],
            "correct_answer": "C"
        },
        {
            "question": "You have fetched JSON data from a REST API using `response = requests.get(url)`. How do you convert this response into a Python dictionary?",
            "options": ["A) response.dict()", "B) response.to_dict()", "C) response.json()", "D) json.loads(response)"],
            "correct_answer": "C"
        },
        {
            "question": "Which pandas function is specifically designed to normalize semi-structured JSON data into a flat DataFrame?",
            "options": ["A) pd.read_json()", "B) pd.json_normalize()", "C) pd.flatten()", "D) pd.DataFrame(json)"],
            "correct_answer": "B"
        },
        {
            "question": "What is the primary purpose of an API key when connecting to a service like Twitter or a stock market feed?",
            "options": ["A) To encrypt the data in transit.", "B) To format the data into XML.", "C) To authenticate the user and track usage limits.", "D) To translate the data into Python objects."],
            "correct_answer": "C"
        },
        {
            "question": "In the `requests` library, what parameter do you use to pass query string arguments (like `?date=2024-01-01`) in a GET request?",
            "options": ["A) data", "B) params", "C) json", "D) headers"],
            "correct_answer": "B"
        },

        # --- BATCH 4: Web Scraping with BeautifulSoup ---
        {
            "question": "Which library is the standard in Python for parsing HTML and XML documents when scraping competitor websites?",
            "options": ["A) Scrapy", "B) urllib", "C) BeautifulSoup", "D) html5lib"],
            "correct_answer": "C"
        },
        {
            "question": "Using BeautifulSoup, how do you find the first `<h1>` tag in a parsed HTML document named `soup`?",
            "options": ["A) soup.find('h1')", "B) soup.get('h1')", "C) soup.extract('h1')", "D) soup.h1_tag()"],
            "correct_answer": "A"
        },
        {
            "question": "You want to scrape all the links from a webpage. Which BeautifulSoup method returns a list of all `<a>` tags?",
            "options": ["A) soup.find_all('a')", "B) soup.search('a')", "C) soup.get_elements('a')", "D) soup.select_links()"],
            "correct_answer": "A"
        },
        {
            "question": "Once you have found an HTML element like `<a href='www.site.com'>Click Here</a>` using BeautifulSoup, how do you extract the text 'Click Here'?",
            "options": ["A) element.content", "B) element.text", "C) element.get_text()", "D) Both B and C"],
            "correct_answer": "D"
        },
        {
            "question": "How do you extract the actual URL (the href attribute) from the BeautifulSoup element `<a href='https://example.com'>Link</a>`?",
            "options": ["A) element.href", "B) element.get('href')", "C) element.attr('href')", "D) element.link()"],
            "correct_answer": "B"
        },

        # --- BATCH 5: SQL Integration in Python ---
        {
            "question": "Which built-in Python library allows you to connect to a lightweight, disk-based SQL database?",
            "options": ["A) psycopg2", "B) sqlalchemy", "C) sqlite3", "D) pyodbc"],
            "correct_answer": "C"
        },
        {
            "question": "After executing an INSERT or UPDATE statement on a SQL database via Python, what method MUST you call to save the changes?",
            "options": ["A) cursor.save()", "B) connection.push()", "C) connection.commit()", "D) cursor.close()"],
            "correct_answer": "C"
        },
        {
            "question": "Which pandas function allows you to execute a SQL query and load the results directly into a DataFrame?",
            "options": ["A) pd.read_sql_query()", "B) pd.sql_to_df()", "C) pd.execute_sql()", "D) pd.fetch_sql()"],
            "correct_answer": "A"
        },
        {
            "question": "What is the purpose of a database `cursor` object in Python?",
            "options": ["A) To point the mouse to the database file.", "B) To execute SQL commands and fetch data from the database.", "C) To encrypt the database connection.", "D) To format the SQL strings."],
            "correct_answer": "B"
        },
        {
            "question": "To prevent SQL injection attacks when querying a database from Python, what should you use?",
            "options": ["A) String concatenation (+)", "B) f-strings", "C) Parameterized queries (using ? or %s)", "D) Regular expressions"],
            "correct_answer": "C"
        },

        # --- BATCH 6: Advanced Text Analytics & String Manipulation ---
        {
            "question": "You have a column of messy text data. Which pandas accessor allows you to apply string methods to an entire Series (e.g., converting to lowercase)?",
            "options": ["A) df['Text'].string.lower()", "B) df['Text'].str.lower()", "C) df['Text'].to_lower()", "D) df['Text'].apply_str('lower')"],
            "correct_answer": "B"
        },
        {
            "question": "In the context of Regular Expressions (regex) imported via the `re` module, what does `re.sub()` do?",
            "options": ["A) Submits a form.", "B) Subtracts numbers found in a string.", "C) Searches for a pattern and replaces it with another string.", "D) Splits a string into substrings."],
            "correct_answer": "C"
        },
        {
            "question": "What regex pattern is commonly used to extract all digits (0-9) from a messy string?",
            "options": ["A) \D+", "B) \d+", "C) \w+", "D) \s+"],
            "correct_answer": "B"
        },
        {
            "question": "Which Natural Language Processing (NLP) library is widely used in Python for tasks like tokenization, lemmatization, and sentiment analysis?",
            "options": ["A) NLTK (Natural Language Toolkit)", "B) BeautifulSoup", "C) Matplotlib", "D) SQLAlchemy"],
            "correct_answer": "A"
        },
        {
            "question": "In text analytics, what are 'stop words'?",
            "options": ["A) Words that cause the Python script to crash.", "B) Punctuation marks at the end of a sentence.", "C) Common words (like 'the', 'is', 'in') that are often removed before analysis to save processing time and improve focus.", "D) Words that signify negative customer sentiment."],
            "correct_answer": "C"
        },

        # --- BATCH 7: Advanced Pandas & Wrangling ---
        {
            "question": "How do you set a specific column, like 'Customer_ID', as the index of a DataFrame `df`?",
            "options": ["A) df.index = 'Customer_ID'", "B) df.set_index('Customer_ID', inplace=True)", "C) df.make_index('Customer_ID')", "D) df.reindex('Customer_ID')"],
            "correct_answer": "B"
        },
        {
            "question": "You need to change the data type of a 'Price' column from integer to float. Which method do you use?",
            "options": ["A) df['Price'].to_float()", "B) df['Price'].astype(float)", "C) df['Price'].convert('float')", "D) float(df['Price'])"],
            "correct_answer": "B"
        },
        {
            "question": "What does the pandas `pd.cut()` function do, which is highly useful for creating customer age brackets?",
            "options": ["A) Deletes specific rows from the dataset.", "B) Segments and sorts continuous data values into discrete bins or buckets.", "C) Trims whitespace from strings.", "D) Halves the size of the DataFrame to save memory."],
            "correct_answer": "B"
        },
        {
            "question": "In pandas, what is a MultiIndex?",
            "options": ["A) A DataFrame with more than 100 columns.", "B) A hierarchical index that allows you to have multiple levels of indexes on a single axis.", "C) An index that uses both letters and numbers.", "D) A separate file that stores index metadata."],
            "correct_answer": "B"
        },
        {
            "question": "You have missing values in a time series dataset (e.g., daily stock prices). Which pandas method fills missing values by propagating the last valid observation forward?",
            "options": ["A) df.fillna(method='ffill')", "B) df.ffill()", "C) Both A and B are correct.", "D) df.fill_forward()"],
            "correct_answer": "C"
        },
        {
            "question": "To randomly sample 10% of your DataFrame rows for a quick audit, which command do you use?",
            "options": ["A) df.sample(frac=0.10)", "B) df.random(0.10)", "C) df.sample(n=10)", "D) df.head(0.10)"],
            "correct_answer": "A"
        },
        {
            "question": "What is the result of `df.isin(['A', 'B'])`?",
            "options": ["A) Deletes rows containing 'A' or 'B'.", "B) Returns a boolean DataFrame showing whether each element is 'A' or 'B'.", "C) Merges two DataFrames named A and B.", "D) Sorts the DataFrame alphabetically."],
            "correct_answer": "B"
        },
        {
            "question": "Which method is best for identifying duplicate rows across an entire DataFrame?",
            "options": ["A) df.duplicates()", "B) df.duplicated()", "C) df.find_dupes()", "D) df.has_duplicates()"],
            "correct_answer": "B"
        },
        {
            "question": "You want to apply a function to a DataFrame that calculates the range (max - min) for each numerical column. Which function paired with `.apply()` is best?",
            "options": ["A) lambda x: x.max() - x.min()", "B) def range(): max - min", "C) df.range()", "D) lambda: max - min"],
            "correct_answer": "A"
        },
        {
            "question": "What does the `pd.get_option('display.max_columns')` command do?",
            "options": ["A) Sets the maximum columns allowed in a DataFrame.", "B) Returns the current setting for how many columns pandas will display in the console or notebook.", "C) Maximizes the width of the columns.", "D) Prints the largest numerical value in each column."],
            "correct_answer": "B"
        },

        # --- BATCH 8: Statistics & Math (SciPy & Statsmodels) ---
        {
            "question": "Which Python library is heavily used alongside pandas for conducting statistical tests (like t-tests and ANOVAs) in business analytics?",
            "options": ["A) Scrapy", "B) SciPy (specifically scipy.stats)", "C) PyTorch", "D) Flask"],
            "correct_answer": "B"
        },
        {
            "question": "In predictive analytics, what does an R-squared value of 0.90 typically indicate about a regression model?",
            "options": ["A) The model is wrong 90% of the time.", "B) 90% of the variance in the dependent variable is predictable from the independent variable(s).", "C) The model has a 90% error rate.", "D) The variables are 90% uncorrelated."],
            "correct_answer": "B"
        },
        {
            "question": "To perform Ordinary Least Squares (OLS) regression with detailed statistical summaries (p-values, standard errors), which library is typically preferred over scikit-learn?",
            "options": ["A) statsmodels", "B) seaborn", "C) numpy", "D) matplotlib"],
            "correct_answer": "A"
        },
        {
            "question": "What is the p-value used for in hypothesis testing?",
            "options": ["A) To predict future values.", "B) To calculate the mean of the sample.", "C) To determine the probability of observing the data given that the null hypothesis is true.", "D) To set the price of a product."],
            "correct_answer": "C"
        },
        {
            "question": "If a p-value is less than your chosen significance level (e.g., 0.05), what is the standard conclusion?",
            "options": ["A) Accept the null hypothesis.", "B) Reject the null hypothesis.", "C) The test failed.", "D) Gather more data immediately."],
            "correct_answer": "B"
        },

        # --- BATCH 9: Advanced Visualization (Plotly & Formatting) ---
        {
            "question": "Unlike Matplotlib and Seaborn, what is the primary advantage of using the Plotly library (`plotly.express`) for business dashboards?",
            "options": ["A) It is built into standard Python.", "B) It generates highly interactive, browser-based plots (hover tooltips, zooming).", "C) It only renders in black and white.", "D) It requires writing JavaScript code manually."],
            "correct_answer": "B"
        },
        {
            "question": "How do you rotate the x-axis labels by 45 degrees in a Matplotlib plot to make long date strings readable?",
            "options": ["A) plt.rotate_x(45)", "B) plt.xticks(rotation=45)", "C) plt.axis('x', angle=45)", "D) plt.xlabel(rotation=45)"],
            "correct_answer": "B"
        },
        {
            "question": "In Seaborn, what parameter do you use to color the data points in a scatterplot based on a categorical variable (e.g., coloring by 'Region')?",
            "options": ["A) color='Region'", "B) fill='Region'", "C) hue='Region'", "D) group='Region'"],
            "correct_answer": "C"
        },
        {
            "question": "Which Matplotlib command clears the current figure, which is essential when generating multiple plots in a loop to avoid overlapping?",
            "options": ["A) plt.clear()", "B) plt.clf()", "C) plt.reset()", "D) plt.empty()"],
            "correct_answer": "B"
        },
        {
            "question": "What does a violin plot (`sns.violinplot`) show that a standard box plot does not?",
            "options": ["A) The median.", "B) The interquartile range.", "C) The full probability density of the data at different values.", "D) Outliers."],
            "correct_answer": "C"
        },

        # --- BATCH 10: Python Core Mastery for Analytics ---
        {
            "question": "What is the output of `list(map(lambda x: x**2, [1, 2, 3]))`?",
            "options": ["A) [1, 2, 3]", "B) [2, 4, 6]", "C) [1, 4, 9]", "D) (1, 4, 9)"],
            "correct_answer": "C"
        },
        {
            "question": "Which Python built-in function returns a list containing only the elements that satisfy a given condition?",
            "options": ["A) map()", "B) reduce()", "C) filter()", "D) apply()"],
            "correct_answer": "C"
        },
        {
            "question": "What is a dictionary comprehension? Which syntax is correct?",
            "options": ["A) {k: v for k, v in iterable}", "B) [k: v for k, v in iterable]", "C) dict(k, v for k, v in iterable)", "D) (k: v for k, v in iterable)"],
            "correct_answer": "A"
        },
        {
            "question": "In Object-Oriented Programming (OOP) in Python, what does the `__init__` method do?",
            "options": ["A) It destroys an object to free memory.", "B) It initializes the attributes of a newly created object (the constructor).", "C) It converts the object to a string.", "D) It imports external libraries."],
            "correct_answer": "B"
        },
        {
            "question": "What is the difference between a Python tuple and a Python list?",
            "options": ["A) Tuples are mutable, lists are immutable.", "B) Lists can hold different data types, tuples cannot.", "C) Tuples are immutable, lists are mutable.", "D) There is no difference."],
            "correct_answer": "C"
        },
        {
            "question": "You have a variable `x = '100'`. Which of the following throws a TypeError?",
            "options": ["A) int(x)", "B) x + '50'", "C) x + 50", "D) type(x)"],
            "correct_answer": "C"
        },
        {
            "question": "What is the purpose of the `pass` statement in Python?",
            "options": ["A) To skip to the next iteration of a loop.", "B) To act as a null operation or placeholder when a statement is syntactically required but you want no code to execute.", "C) To bypass an error.", "D) To pass a variable to a function."],
            "correct_answer": "B"
        },
        {
            "question": "Which operator is used to unpack a dictionary into keyword arguments when calling a function? (e.g., `func(***my_dict)` )",
            "options": ["A) *", "B) &", "C) **", "D) //"],
            "correct_answer": "C"
        },
        {
            "question": "What does the `set()` function do when applied to a list containing duplicate values?",
            "options": ["A) Sorts the list.", "B) Removes all duplicate values, leaving only unique elements.", "C) Reverses the list.", "D) Throws an error."],
            "correct_answer": "B"
        },
        {
            "question": "Which of the following is considered 'Pythonic' way to swap the values of variables `a` and `b`?",
            "options": ["A) temp = a; a = b; b = temp", "B) a, b = b, a", "C) swap(a, b)", "D) a = b; b = a"],
            "correct_answer": "B"
        },
        
        # --- BATCH 11: Extra Time-Series & Logistics ---
        {
            "question": "In pandas, what does `pd.date_range(start='2024-01-01', periods=5, freq='D')` generate?",
            "options": ["A) A list of 5 random dates in 2024.", "B) A DatetimeIndex containing 5 consecutive days starting from Jan 1, 2024.", "C) A single string with 5 dates.", "D) An error."],
            "correct_answer": "B"
        },
        {
            "question": "How do you calculate the percentage change between the current and a prior element in a pandas Series (highly useful for stock returns)?",
            "options": ["A) s.pct_change()", "B) s.percentage()", "C) s.diff() / s", "D) s.growth()"],
            "correct_answer": "A"
        },
        {
            "question": "If `df.corr()` shows a correlation of -0.85 between 'Price' and 'Sales_Volume', what does this mean?",
            "options": ["A) A strong positive relationship.", "B) A weak negative relationship.", "C) A strong negative relationship (as Price goes up, Volume goes down significantly).", "D) No relationship."],
            "correct_answer": "C"
        },
        {
            "question": "When defining a machine learning pipeline, what is the problem with 'Data Leakage'?",
            "options": ["A) When you run out of memory (RAM).", "B) When information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates.", "C) When missing values are accidentally deleted.", "D) When a database connection drops."],
            "correct_answer": "B"
        },
        {
            "question": "What does the scikit-learn module `GridSearchCV` do?",
            "options": ["A) Creates a grid layout for Matplotlib.", "B) Exhaustively searches over specified parameter values for an estimator to find the best hyperparameters.", "C) Performs K-Means clustering.", "D) Scrapes data from HTML tables."],
            "correct_answer": "B"
        },
        {
            "question": "In forecasting, what does ARIMA stand for?",
            "options": ["A) Automated Regression Integrated Moving Average", "B) Auto-Regressive Integrated Moving Average", "C) Advanced Regional Indexing Matrix", "D) Artificial Relational Integration Model"],
            "correct_answer": "B"
        },
        {
            "question": "Which Python library is the industry standard for creating standalone web applications and APIs directly from your analytics scripts?",
            "options": ["A) Pandas", "B) Seaborn", "C) Flask (or FastAPI)", "D) SQLAlchemy"],
            "correct_answer": "C"
        },
        {
            "question": "You want to round all values in a pandas DataFrame to 2 decimal places. What is the command?",
            "options": ["A) df.round(2)", "B) df.set_decimals(2)", "C) df.format(2)", "D) df.limit(2)"],
            "correct_answer": "A"
        },
        {
            "question": "When creating a decision tree classifier, what does the 'max_depth' parameter control?",
            "options": ["A) The number of trees in the forest.", "B) The maximum number of levels the tree is allowed to grow, helping to prevent overfitting.", "C) The amount of data used for testing.", "D) The color depth of the plotted tree."],
            "correct_answer": "B"
        },
        {
            "question": "To read an Excel file with multiple tabs into pandas, which parameter do you use to specify the tab name?",
            "options": ["A) tab_name", "B) sheet_name", "C) page", "D) index"],
            "correct_answer": "B"
        },
        # --- BATCH 12: A/B Testing & Advanced SciPy ---
        {
            "question": "When conducting an A/B test on two website designs, which statistical test is most commonly used to compare the conversion rates (proportions) of the two groups?",
            "options": ["A) T-test", "B) Chi-Square Test", "C) ANOVA", "D) Pearson Correlation"],
            "correct_answer": "B"
        },
        {
            "question": "Using `scipy.stats`, which function performs an independent two-sample t-test to compare the average order value of two different customer segments?",
            "options": ["A) stats.ttest_ind()", "B) stats.ttest_rel()", "C) stats.ttest_1samp()", "D) stats.anova()"],
            "correct_answer": "A"
        },
        {
            "question": "What does a Type I error represent in the context of an A/B test?",
            "options": ["A) Failing to reject a false null hypothesis (False Negative).", "B) Rejecting a true null hypothesis (False Positive).", "C) A bug in the Python code.", "D) Incorrectly collecting the data."],
            "correct_answer": "B"
        },
        {
            "question": "How do you calculate the variance of a pandas Series `s`?",
            "options": ["A) s.var()", "B) s.variance()", "C) s.std() ** 2", "D) Both A and C are correct."],
            "correct_answer": "D"
        },
        {
            "question": "Which SciPy function is used to find the Pearson correlation coefficient and the p-value for testing non-correlation?",
            "options": ["A) scipy.stats.pearsonr()", "B) scipy.stats.corr()", "C) scipy.stats.spearmanr()", "D) scipy.stats.kendalltau()"],
            "correct_answer": "A"
        },

        # --- BATCH 13: Dimensionality Reduction & PCA ---
        {
            "question": "What is the primary goal of Principal Component Analysis (PCA) in machine learning?",
            "options": ["A) To increase the number of features.", "B) To reduce the dimensionality of a dataset while retaining as much variance as possible.", "C) To classify categorical variables.", "D) To handle missing values automatically."],
            "correct_answer": "B"
        },
        {
            "question": "Before applying PCA using scikit-learn, what vital preprocessing step must you perform on your features?",
            "options": ["A) Convert everything to strings.", "B) Standardize or scale the data (e.g., StandardScaler).", "C) Remove all outliers.", "D) Multiply all values by 100."],
            "correct_answer": "B"
        },
        {
            "question": "In scikit-learn's PCA, which attribute tells you how much information (variance) is attributed to each of the principal components?",
            "options": ["A) pca.components_", "B) pca.explained_variance_ratio_", "C) pca.variance_", "D) pca.info_"],
            "correct_answer": "B"
        },
        {
            "question": "If the first two components of a PCA model explain 95% of the variance, what does this imply?",
            "options": ["A) The model is 95% accurate.", "B) You can safely drop the remaining components and still retain 95% of the original data's information.", "C) 95% of the data is useless.", "D) The model is overfitting."],
            "correct_answer": "B"
        },
        {
            "question": "Which algorithm is a popular non-linear dimensionality reduction technique often used for visualizing high-dimensional data in 2D or 3D?",
            "options": ["A) Linear Regression", "B) t-SNE", "C) K-Means", "D) Naive Bayes"],
            "correct_answer": "B"
        },

        # --- BATCH 14: PySpark & Big Data Concepts ---
        {
            "question": "What is the primary data structure in PySpark that is similar to a pandas DataFrame but distributed across a cluster?",
            "options": ["A) RDD (Resilient Distributed Dataset)", "B) Spark DataFrame", "C) Dask Array", "D) Distributed Matrix"],
            "correct_answer": "B"
        },
        {
            "question": "In PySpark, transformations (like `.filter()` or `.select()`) are 'lazy'. What does this mean?",
            "options": ["A) They take a long time to run.", "B) They are not executed until an 'action' (like `.show()` or `.count()`) is called.", "C) They use very little memory.", "D) They only work on small datasets."],
            "correct_answer": "B"
        },
        {
            "question": "How do you start a Spark session in a PySpark script?",
            "options": ["A) spark = SparkSession.builder.getOrCreate()", "B) spark = pd.Spark()", "C) spark = init_spark()", "D) spark = start_cluster()"],
            "correct_answer": "A"
        },
        {
            "question": "Which PySpark DataFrame method is equivalent to pandas `df.head()` for viewing the first few rows?",
            "options": ["A) df.show()", "B) df.top()", "C) df.view()", "D) df.display()"],
            "correct_answer": "A"
        },
        {
            "question": "If a dataset is too large to fit in memory on a single machine, which Python library is an alternative to pandas for parallel computing?",
            "options": ["A) NumPy", "B) Dask", "C) Seaborn", "D) Statsmodels"],
            "correct_answer": "B"
        },

        # --- BATCH 15: Advanced Groupby & Aggregations ---
        {
            "question": "You want to group by 'Region' and apply different aggregations to different columns (e.g., sum of 'Sales', mean of 'Profit'). How do you do this in pandas?",
            "options": ["A) df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'mean'})", "B) df.groupby('Region')['Sales', 'Profit'].apply('sum', 'mean')", "C) df.groupby('Region').aggregate('Sales'='sum', 'Profit'='mean')", "D) df.group('Region').multi_agg()"],
            "correct_answer": "A"
        },
        {
            "question": "What does the `.transform()` function do when chained after a `.groupby()`?",
            "options": ["A) It changes the data types of the columns.", "B) It returns an aggregated DataFrame with a new shape.", "C) It returns a Series or DataFrame with the same shape as the original, broadcasting the aggregated values back to the original rows.", "D) It transposes the rows and columns."],
            "correct_answer": "C"
        },
        {
            "question": "How do you calculate a rolling 30-day sum of 'Revenue' in pandas?",
            "options": ["A) df['Revenue'].rolling(window=30).sum()", "B) df['Revenue'].roll(30).add()", "C) df['Revenue'].moving_sum(30)", "D) df['Revenue'].sum(window=30)"],
            "correct_answer": "A"
        },
        {
            "question": "Which pandas method allows you to bin numerical data into discrete intervals based on quantiles (e.g., creating 4 equal-sized customer quartiles)?",
            "options": ["A) pd.cut()", "B) pd.qcut()", "C) pd.bin()", "D) pd.split()"],
            "correct_answer": "B"
        },
        {
            "question": "When working with hierarchical indexes (MultiIndex), how do you select data from the inner level?",
            "options": ["A) df.loc[:, 'inner_label']", "B) df.xs('inner_label', level=1)", "C) df.index(level=1)", "D) Both A and B"],
            "correct_answer": "B"
        },

        # --- BATCH 16: Recommender Systems & Distance Metrics ---
        {
            "question": "In building a recommendation engine, which metric is commonly used to measure the similarity between two user's rating vectors?",
            "options": ["A) Euclidean Distance", "B) Cosine Similarity", "C) Manhattan Distance", "D) All of the above"],
            "correct_answer": "D"
        },
        {
            "question": "What does Cosine Similarity measure?",
            "options": ["A) The physical distance between two points.", "B) The cosine of the angle between two non-zero vectors, indicating their orientation regardless of magnitude.", "C) The absolute difference between values.", "D) The statistical significance of a correlation."],
            "correct_answer": "B"
        },
        {
            "question": "In Collaborative Filtering, what is the 'Cold Start' problem?",
            "options": ["A) When the Python kernel takes too long to load.", "B) When the system cannot draw any inferences for users or items about which it has not yet gathered sufficient information.", "C) When the server crashes due to high traffic.", "D) When the algorithm gets stuck in a local minimum."],
            "correct_answer": "B"
        },
        {
            "question": "Which matrix operation is fundamental to Latent Factor Models in recommendation systems?",
            "options": ["A) Matrix Addition", "B) Singular Value Decomposition (SVD)", "C) Matrix Inversion", "D) Cross Product"],
            "correct_answer": "B"
        },
        {
            "question": "In Market Basket Analysis, what does the 'Support' metric indicate for an itemset {A, B}?",
            "options": ["A) How often A and B are purchased together out of all transactions.", "B) The likelihood of buying B given A is bought.", "C) The profit margin of A and B.", "D) The total revenue from A and B."],
            "correct_answer": "A"
        },

        # --- BATCH 17: Machine Learning - Classification & Regression ---
        {
            "question": "Which algorithm uses the concept of 'hyperplanes' to separate data classes in a high-dimensional space?",
            "options": ["A) Decision Trees", "B) Support Vector Machines (SVM)", "C) Logistic Regression", "D) K-Nearest Neighbors"],
            "correct_answer": "B"
        },
        {
            "question": "In K-Nearest Neighbors (KNN), what happens if you choose a 'K' value that is too small (e.g., K=1)?",
            "options": ["A) The model will underfit the data.", "B) The model becomes highly sensitive to noise and will overfit.", "C) The algorithm will run much slower.", "D) The model cannot be trained."],
            "correct_answer": "B"
        },
        {
            "question": "What is the purpose of cross-validation (e.g., K-Fold Cross-Validation)?",
            "options": ["A) To test the model on multiple different datasets from the internet.", "B) To validate that the code has no syntax errors.", "C) To ensure the model's performance is robust and not dependent on a single random train/test split.", "D) To increase the speed of the training process."],
            "correct_answer": "C"
        },
        {
            "question": "What does a Confusion Matrix display for a binary classification problem?",
            "options": ["A) The R-squared and MSE values.", "B) The True Positives, True Negatives, False Positives, and False Negatives.", "C) The correlations between all features.", "D) The hyperparameters of the model."],
            "correct_answer": "B"
        },
        {
            "question": "In a Random Forest, what is 'Feature Importance'?",
            "options": ["A) A rule that you must scale features before training.", "B) A score indicating how useful or valuable each feature was in the construction of the decision trees within the model.", "C) The label you are trying to predict.", "D) The statistical p-value of the features."],
            "correct_answer": "B"
        },

        # --- BATCH 18: Python Performance & Optimization ---
        {
            "question": "Why is iterating over a pandas DataFrame using a standard `for` loop (e.g., `for index, row in df.iterrows():`) discouraged in business analytics?",
            "options": ["A) It is syntactically invalid.", "B) It is extremely slow compared to vectorized operations.", "C) It deletes the data as it loops.", "D) It requires excessive RAM."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'vectorization' in NumPy and pandas?",
            "options": ["A) Converting text to SVG vector graphics.", "B) Applying operations to entire arrays or series at once, leveraging optimized C-level code, rather than looping item by item.", "C) Using machine learning to predict vectors.", "D) Sorting data by column headers."],
            "correct_answer": "B"
        },
        {
            "question": "Which Python library is explicitly designed to profile your code and tell you exactly how much time is spent executing each function?",
            "options": ["A) timeit", "B) cProfile", "C) pandas_profiling", "D) memory_profiler"],
            "correct_answer": "B"
        },
        {
            "question": "When loading a massive 10GB CSV file in pandas, which parameter allows you to read it in smaller, manageable pieces to avoid crashing your RAM?",
            "options": ["A) split=True", "B) pieces=1000", "C) chunksize=10000", "D) max_rows=1000"],
            "correct_answer": "C"
        },
        {
            "question": "What is the most memory-efficient way to store categorical string data (like 'Low', 'Medium', 'High') in pandas?",
            "options": ["A) Keep it as the default 'object' type.", "B) Convert it to 'category' data type using `astype('category')`.", "C) Convert it to 'string'.", "D) Convert it to boolean."],
            "correct_answer": "B"
        },

        # --- BATCH 19: Python Core - Decorators & Context Managers ---
        {
            "question": "What is a decorator in Python (indicated by the `@` symbol)?",
            "options": ["A) A styling tool for Matplotlib.", "B) A function that takes another function and extends its behavior without explicitly modifying it.", "C) A class method.", "D) A variable that holds an HTML tag."],
            "correct_answer": "B"
        },
        {
            "question": "What does the `with` statement (a context manager) do when opening a file in Python (`with open('file.txt') as f:`)?",
            "options": ["A) It automatically encrypts the file.", "B) It ensures the file is properly closed after the block of code executes, even if an error occurs.", "C) It prevents other users from reading the file.", "D) It deletes the file after reading."],
            "correct_answer": "B"
        },
        {
            "question": "What is the difference between `==` and `is` in Python?",
            "options": ["A) They are identical.", "B) `==` checks for value equality, while `is` checks for object identity (if they point to the exact same memory location).", "C) `is` checks for value equality, while `==` checks for object identity.", "D) `is` is used for strings, `==` is used for numbers."],
            "correct_answer": "B"
        },
        {
            "question": "If you want a variable inside a function to modify a variable declared outside the function, which keyword must you use?",
            "options": ["A) external", "B) global", "C) static", "D) public"],
            "correct_answer": "B"
        },
        {
            "question": "What does the `__name__ == \"__main__\"` block do in a Python script?",
            "options": ["A) It forces the script to run as an administrator.", "B) It ensures the code block only runs if the script is executed directly, not if it is imported as a module by another script.", "C) It defines the main function.", "D) It renames the file."],
            "correct_answer": "B"
        },

        # --- BATCH 20: Advanced Web Scraping & Automation ---
        {
            "question": "If a website relies heavily on JavaScript to load its data dynamically, BeautifulSoup will not work out-of-the-box. Which library should you use instead to automate a real browser?",
            "options": ["A) Requests", "B) Selenium", "C) Scrapy", "D) Pandas"],
            "correct_answer": "B"
        },
        {
            "question": "In Selenium, how do you locate an input box on a webpage using its HTML ID?",
            "options": ["A) driver.find_element(By.ID, 'search-box')", "B) driver.get_id('search-box')", "C) driver.locate('search-box')", "D) driver.search_id('search-box')"],
            "correct_answer": "A"
        },
        {
            "question": "What is the purpose of `time.sleep(5)` when scraping a website?",
            "options": ["A) To pause the script for 5 seconds to avoid overloading the server or getting blocked.", "B) To make the internet connection faster.", "C) To shut down the computer after 5 minutes.", "D) To skip 5 rows of data."],
            "correct_answer": "A"
        },
        {
            "question": "What is an HTML User-Agent?",
            "options": ["A) A human user.", "B) A string sent in the HTTP header telling the server what browser and operating system the requester is using.", "C) A firewall that blocks scraping.", "D) The IP address of the server."],
            "correct_answer": "B"
        },
        {
            "question": "Which pandas method allows you to instantly scrape all HTML `<table>` tags on a webpage and convert them directly into DataFrames?",
            "options": ["A) pd.read_html()", "B) pd.read_tables()", "C) pd.scrape_web()", "D) pd.get_tables()"],
            "correct_answer": "A"
        },

        # --- BATCH 21: Deep Learning Basics (TensorFlow/Keras) ---
        {
            "question": "In a neural network, what is an 'Epoch'?",
            "options": ["A) A single layer of neurons.", "B) One complete pass of the entire training dataset through the algorithm.", "C) The error rate of the model.", "D) The activation function."],
            "correct_answer": "B"
        },
        {
            "question": "Which activation function is most commonly used in the hidden layers of a modern deep neural network?",
            "options": ["A) Sigmoid", "B) Tanh", "C) ReLU (Rectified Linear Unit)", "D) Softmax"],
            "correct_answer": "C"
        },
        {
            "question": "If you are building a multi-class classification neural network (e.g., predicting 5 different customer segments), which activation function should be on the final output layer?",
            "options": ["A) ReLU", "B) Linear", "C) Sigmoid", "D) Softmax"],
            "correct_answer": "D"
        },
        {
            "question": "What is 'Dropout' in the context of deep learning?",
            "options": ["A) When the model stops training early.", "B) A regularization technique where randomly selected neurons are ignored during training to prevent overfitting.", "C) Deleting missing values from the dataset.", "D) The learning rate decaying to zero."],
            "correct_answer": "B"
        },
        {
            "question": "Which optimization algorithm is currently the most popular choice for training neural networks due to its adaptive learning rate?",
            "options": ["A) Standard Gradient Descent", "B) Adam", "C) RMSprop", "D) Adagrad"],
            "correct_answer": "B"
        },
        # --- BATCH 22: Advanced Time-Series & Forecasting ---
        {
            "question": "When using Facebook's Prophet library for time-series forecasting, what are the mandatory names for the date and target columns in your pandas DataFrame?",
            "options": ["A) 'Date' and 'Value'", "B) 'ds' and 'y'", "C) 'Time' and 'Target'", "D) 'x' and 'y'"],
            "correct_answer": "B"
        },
        {
            "question": "What is the primary advantage of Prophet over traditional ARIMA models for business analysts?",
            "options": ["A) It handles missing data and large outliers extremely well automatically.", "B) It processes image data.", "C) It only requires 3 data points to make an accurate 10-year forecast.", "D) It is built directly into pandas."],
            "correct_answer": "A"
        },
        {
            "question": "In statsmodels, if you are conducting an Augmented Dickey-Fuller (ADF) test, what are you testing your time-series data for?",
            "options": ["A) Normality", "B) Stationarity (whether statistical properties are constant over time)", "C) Homoscedasticity", "D) Collinearity"],
            "correct_answer": "B"
        },
        {
            "question": "If an ADF test yields a p-value of 0.85, what should you typically do before fitting an ARIMA model?",
            "options": ["A) Fit the model immediately.", "B) Difference the data (e.g., `df.diff()`) because the series is non-stationary.", "C) Convert the data to strings.", "D) Drop all negative values."],
            "correct_answer": "B"
        },
        {
            "question": "What does the 'seasonality' component of a time-series decomposition represent?",
            "options": ["A) The overall long-term direction of the data.", "B) The random, unpredictable noise.", "C) Repeating, predictable patterns at fixed intervals (e.g., higher sales every December).", "D) The impact of sudden market crashes."],
            "correct_answer": "C"
        },

        # --- BATCH 23: Prescriptive Analytics & Optimization ---
        {
            "question": "Which Python library is the standard for solving Linear Programming problems (like supply chain optimization) in business analytics?",
            "options": ["A) Seaborn", "B) PuLP", "C) Beautiful Soup", "D) NLTK"],
            "correct_answer": "B"
        },
        {
            "question": "In a linear programming model, what is the 'Objective Function'?",
            "options": ["A) The Python script that loads the data.", "B) The mathematical equation you are trying to maximize or minimize (e.g., maximizing profit or minimizing cost).", "C) The limits on your resources.", "D) The final output visualization."],
            "correct_answer": "B"
        },
        {
            "question": "When using `scipy.optimize.linprog`, how does the algorithm handle maximization problems by default?",
            "options": ["A) It maximizes automatically.", "B) You must multiply your objective function coefficients by -1, as it only minimizes by default.", "C) It asks the user for input during execution.", "D) It cannot solve maximization problems."],
            "correct_answer": "B"
        },
        {
            "question": "What are 'Constraints' in the context of prescriptive analytics optimization?",
            "options": ["A) The RAM limitations of your computer.", "B) Mathematical limits restricting your decision variables (e.g., you cannot produce more than 500 units due to factory capacity).", "C) The time it takes for a machine learning model to train.", "D) The firewall blocking database access."],
            "correct_answer": "B"
        },
        {
            "question": "What does a 'Shadow Price' (or Dual Value) tell a business analyst in a linear programming solution?",
            "options": ["A) The black market value of a product.", "B) The amount the objective function would improve if the constraint was relaxed by one unit.", "C) The hidden tax implications.", "D) The cost of running the algorithm."],
            "correct_answer": "B"
        },

        # --- BATCH 24: Advanced ML Pipelines & ColumnTransformers ---
        {
            "question": "In scikit-learn, what is the purpose of a `Pipeline`?",
            "options": ["A) To chain multiple data preprocessing steps and an estimator into a single, cohesive object.", "B) To download data from the cloud.", "C) To connect pandas to SQL.", "D) To stream video data."],
            "correct_answer": "A"
        },
        {
            "question": "Why is using a `Pipeline` crucial when performing Cross-Validation?",
            "options": ["A) It prevents data leakage by ensuring preprocessing (like scaling) is fitted ONLY on the training folds, not the entire dataset.", "B) It makes the code look more colorful.", "C) It uses less CPU power.", "D) It automatically converts Python to C++."],
            "correct_answer": "A"
        },
        {
            "question": "What does `ColumnTransformer` allow you to do in a scikit-learn preprocessing workflow?",
            "options": ["A) Delete columns randomly.", "B) Apply different preprocessing steps to different columns (e.g., OneHotEncode categorical columns while StandardScaling numerical columns).", "C) Rename columns automatically.", "D) Merge two DataFrames."],
            "correct_answer": "B"
        },
        {
            "question": "Which of the following creates an instance of a scikit-learn Pipeline with a standard scaler followed by a logistic regression model?",
            "options": ["A) Pipeline(StandardScaler(), LogisticRegression())", "B) Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])", "C) Pipeline(steps=[StandardScaler, LogisticRegression])", "D) make_pipeline('StandardScaler', 'LogisticRegression')"],
            "correct_answer": "B"
        },
        {
            "question": "How do you access the coefficients (weights) of a LinearRegression model that is the last step in a fitted Pipeline named `pipe`?",
            "options": ["A) pipe.coef_", "B) pipe.named_steps['model_name'].coef_", "C) pipe.get_weights()", "D) pipe[-1].weights_"],
            "correct_answer": "B"
        },

        # --- BATCH 25: Advanced NLP (spaCy & Text Mining) ---
        {
            "question": "While NLTK is great for academic text processing, which Python NLP library is heavily favored in business production environments for its speed and object-oriented approach?",
            "options": ["A) Matplotlib", "B) spaCy", "C) Requests", "D) PyTorch"],
            "correct_answer": "B"
        },
        {
            "question": "What is Named Entity Recognition (NER) in text analytics?",
            "options": ["A) Counting the number of words in a document.", "B) Identifying and classifying key information in text into predefined categories like Person, Organization, Location, or Monetary Value.", "C) Translating English to French.", "D) Fixing spelling errors."],
            "correct_answer": "B"
        },
        {
            "question": "What does 'Lemmatization' do to a word in natural language processing?",
            "options": ["A) Capitalizes it.", "B) Reduces the word to its base or dictionary form (e.g., 'running' becomes 'run').", "C) Deletes the word if it is a vowel.", "D) Translates it to binary."],
            "correct_answer": "B"
        },
        {
            "question": "In the context of the `CountVectorizer` in scikit-learn, what does converting text to a 'Bag of Words' mean?",
            "options": ["A) Shuffling the order of the words randomly.", "B) Representing a document as a matrix of word frequencies, completely ignoring word order and grammar.", "C) Zipping text files to save storage.", "D) Using a dictionary to look up definitions."],
            "correct_answer": "B"
        },
        {
            "question": "What does TF-IDF stand for in text analytics?",
            "options": ["A) Text Frequency - Inverse Document Format", "B) Term Frequency - Inverse Document Frequency", "C) Total Finds - Individual Data Fields", "D) Text Formatting - Index Data Flow"],
            "correct_answer": "B"
        },

        # --- BATCH 26: Cloud Integrations & Boto3 ---
        {
            "question": "Which Python library acts as the official Amazon Web Services (AWS) SDK, allowing you to manage S3 buckets and EC2 instances directly from Python?",
            "options": ["A) awspy", "B) cloud_connect", "C) boto3", "D) s3_manager"],
            "correct_answer": "C"
        },
        {
            "question": "In AWS architecture accessed via Python, what is Amazon S3 primarily used for?",
            "options": ["A) Running SQL databases.", "B) Object storage (storing large flat files like CSVs, images, or JSON).", "C) Training neural networks.", "D) Hosting interactive websites."],
            "correct_answer": "B"
        },
        {
            "question": "When querying a massive dataset sitting in an S3 bucket without loading the whole file into pandas, which AWS service is commonly queried via Python using `awswrangler` or `boto3`?",
            "options": ["A) Amazon Athena", "B) Amazon Lex", "C) Amazon Route53", "D) Amazon CloudFront"],
            "correct_answer": "A"
        },
        {
            "question": "How do you securely pass cloud credentials (like AWS Access Keys) to your Python script without hardcoding them into the file?",
            "options": ["A) Print them at the top of the file.", "B) Use environment variables (e.g., `os.environ`) or configuration files (e.g., `.env`).", "C) Store them in a public GitHub repository.", "D) Hardcode them as comments."],
            "correct_answer": "B"
        },
        {
            "question": "Which library is the Google Cloud Platform (GCP) equivalent of boto3, used for interacting with services like BigQuery?",
            "options": ["A) gcloud-python", "B) google-cloud-bigquery", "C) gcp_sdk", "D) bq_python"],
            "correct_answer": "B"
        },

        # --- BATCH 27: Advanced SQL Window Functions in Python ---
        {
            "question": "When executing a SQL query via Python, what does the `OVER()` clause signify?",
            "options": ["A) The query is finished.", "B) It defines a Window Function, allowing you to perform calculations across a set of table rows related to the current row.", "C) It limits the results to 10 rows.", "D) It drops the table."],
            "correct_answer": "B"
        },
        {
            "question": "If you use `pd.read_sql(\"SELECT Employee, Salary, RANK() OVER (ORDER BY Salary DESC) FROM Employees\", conn)`, what is the result?",
            "options": ["A) A DataFrame sorted by Employee name.", "B) A DataFrame assigning a rank to each employee based on their salary, with the highest salary ranked 1.", "C) An error.", "D) A DataFrame with only the highest-paid employee."],
            "correct_answer": "B"
        },
        {
            "question": "Which SQL keyword is used within a Window Function to calculate a running total partitioned by a specific category (e.g., monthly sales per region)?",
            "options": ["A) PARTITION BY", "B) GROUP BY", "C) ORDER BY", "D) SEGMENT BY"],
            "correct_answer": "A"
        },
        {
            "question": "What is the primary difference between `GROUP BY` and a Window Function?",
            "options": ["A) Window functions are faster.", "B) `GROUP BY` collapses the rows into a single summary row, while a Window Function preserves the original number of rows while appending the aggregated calculation.", "C) There is no difference.", "D) Window functions only work on numeric data."],
            "correct_answer": "B"
        },
        {
            "question": "Which Python ORM (Object-Relational Mapper) allows you to write SQL queries using pure Python classes and objects?",
            "options": ["A) SQLAlchemy", "B) SQLite3", "C) PyODBC", "D) Psycopg2"],
            "correct_answer": "A"
        },

        # --- BATCH 28: Network Analysis & Graph Theory ---
        {
            "question": "Which Python library is the industry standard for studying complex networks, graphs, and supply chain routing?",
            "options": ["A) Pandas", "B) NetworkX", "C) Matplotlib", "D) Keras"],
            "correct_answer": "B"
        },
        {
            "question": "In a NetworkX graph, what do 'Nodes' and 'Edges' represent in a social network context?",
            "options": ["A) Nodes are connections, Edges are people.", "B) Nodes are people, Edges are the connections (friendships) between them.", "C) Nodes are servers, Edges are IP addresses.", "D) Nodes are algorithms, Edges are outputs."],
            "correct_answer": "B"
        },
        {
            "question": "What does the 'Degree Centrality' of a node measure?",
            "options": ["A) The temperature of the server.", "B) The fraction of nodes it is directly connected to (e.g., the person with the most direct friends).", "C) The shortest path to the end of the network.", "D) The geographical location of the node."],
            "correct_answer": "B"
        },
        {
            "question": "If you are optimizing delivery routes, which algorithm finds the most efficient path between two nodes in a weighted graph?",
            "options": ["A) Random Forest", "B) Dijkstra's Algorithm", "C) K-Means", "D) Naive Bayes"],
            "correct_answer": "B"
        },
        {
            "question": "What is a 'Directed Graph' (DiGraph) in NetworkX?",
            "options": ["A) A graph where all connections go in both directions equally.", "B) A graph where edges have a specific direction (e.g., Twitter followers, where A follows B, but B does not necessarily follow A).", "C) A graph managed by a director.", "D) A 3D graph."],
            "correct_answer": "B"
        },

        # --- BATCH 29: Simulation & Monte Carlo Methods ---
        {
            "question": "What is a Monte Carlo simulation in business analytics?",
            "options": ["A) A gambling algorithm.", "B) A technique used to understand the impact of risk and uncertainty in financial forecasting by running thousands of randomized scenarios.", "C) A method to clean text data.", "D) A visualization tool."],
            "correct_answer": "B"
        },
        {
            "question": "Which NumPy module is heavily relied upon to generate the randomized variables needed for a Monte Carlo simulation?",
            "options": ["A) np.linalg", "B) np.random", "C) np.fft", "D) np.matrix"],
            "correct_answer": "B"
        },
        {
            "question": "If you want to simulate daily stock returns based on a normal distribution with a specific mean and standard deviation, which function do you use?",
            "options": ["A) np.random.normal(loc=mean, scale=std, size=days)", "B) np.random.uniform()", "C) np.random.choice()", "D) np.random.poisson()"],
            "correct_answer": "A"
        },
        {
            "question": "When simulating customer arrivals at a store (events happening in a fixed interval of time), which probability distribution is most appropriate to model the data?",
            "options": ["A) Normal Distribution", "B) Binomial Distribution", "C) Poisson Distribution", "D) Uniform Distribution"],
            "correct_answer": "C"
        },
        {
            "question": "What is the Law of Large Numbers, which underpins the logic of Monte Carlo simulations?",
            "options": ["A) Large numbers cause Python to crash.", "B) As the number of trials increases, the average of the results will converge on the expected, true theoretical value.", "C) You must use a 64-bit operating system.", "D) More data always equals more profit."],
            "correct_answer": "B"
        },

        # --- BATCH 30: MLOps, Version Control & Model Deployment ---
        {
            "question": "What is the primary purpose of the `MLflow` library in a business analytics team?",
            "options": ["A) To build neural networks.", "B) To manage the end-to-end machine learning lifecycle, including tracking experiments, recording parameters, and saving model versions.", "C) To scrape data from the web.", "D) To create interactive dashboards."],
            "correct_answer": "B"
        },
        {
            "question": "You trained a highly accurate Random Forest model and want to save it to your local hard drive so you can use it tomorrow without retraining. Which library do you use?",
            "options": ["A) pandas", "B) joblib (or pickle)", "C) requests", "D) matplotlib"],
            "correct_answer": "B"
        },
        {
            "question": "What is Docker used for in the context of deploying Python analytics applications?",
            "options": ["A) It creates a lightweight, portable container that packages your code and all its dependencies, ensuring it runs exactly the same on any computer or server.", "B) It connects to SQL databases.", "C) It improves the visual aesthetics of the app.", "D) It compresses files."],
            "correct_answer": "A"
        },
        {
            "question": "In Git version control, what does the `git commit -m \"message\"` command do?",
            "options": ["A) Uploads code to the cloud.", "B) Downloads code from GitHub.", "C) Saves a snapshot of your currently staged changes to your local repository history.", "D) Deletes the code."],
            "correct_answer": "C"
        },
        {
            "question": "When deploying a Python REST API using FastAPI or Flask, what format is typically used to send the model's predictions back to the user or front-end application?",
            "options": ["A) A raw CSV file", "B) A Python dictionary", "C) JSON (JavaScript Object Notation)", "D) XML"],
            "correct_answer": "C"
        },

        # --- BATCH 31: Advanced Aggregations & Python Built-ins ---
        {
            "question": "Which Python module provides specialized container datatypes like `Counter`, `defaultdict`, and `namedtuple`?",
            "options": ["A) math", "B) collections", "C) itertools", "D) sys"],
            "correct_answer": "B"
        },
        {
            "question": "If you want to count the frequency of every word in a list instantly without writing a loop, what should you use?",
            "options": ["A) The collections.Counter() class", "B) A pandas DataFrame", "C) The math.factorial() function", "D) A regular expression"],
            "correct_answer": "A"
        },
        {
            "question": "What is the primary advantage of using a Python `set` over a `list` when checking if an item exists (`if item in collection:`)?",
            "options": ["A) Sets preserve the order of elements.", "B) Checking membership in a set is exceptionally fast (O(1) time complexity) compared to a list.", "C) Sets use more memory.", "D) Sets can hold mutable objects like dictionaries."],
            "correct_answer": "B"
        },
        {
            "question": "In pandas, what does the `.explode()` method do?",
            "options": ["A) Deletes the DataFrame.", "B) Transforms each element of a list-like column into its own separate row, replicating index values.", "C) Separates a string column into multiple columns.", "D) Merges all rows into one."],
            "correct_answer": "B"
        },
        {
            "question": "You are analyzing website traffic logs and need to calculate the difference in days between a user's 'First_Visit' and 'Last_Visit'. Which accessor is required?",
            "options": ["A) df['Last_Visit'].str - df['First_Visit'].str", "B) (df['Last_Visit'] - df['First_Visit']).dt.days", "C) df['Last_Visit'].time() - df['First_Visit'].time()", "D) diff(df['Last_Visit'], df['First_Visit'])"],
            "correct_answer": "B"
        },
        
        # --- BATCH 32: Extra Master's Level Business Logic ---
        {
            "question": "What is 'Feature Engineering' in machine learning?",
            "options": ["A) Upgrading the computer hardware.", "B) The process of using domain knowledge to create new input variables (features) from raw data to improve model accuracy.", "C) Writing the algorithm in C++ for speed.", "D) Deleting the target variable."],
            "correct_answer": "B"
        },
        {
            "question": "When analyzing inventory data, what does the formula `np.where(df['Stock'] < 10, 'Reorder', 'Sufficient')` do?",
            "options": ["A) Drops rows where Stock is less than 10.", "B) Returns a new array where values are 'Reorder' if the condition is met, and 'Sufficient' otherwise.", "C) Raises an error.", "D) Changes all stock values to 10."],
            "correct_answer": "B"
        },
        {
            "question": "What is the 'Curse of Dimensionality' in data science?",
            "options": ["A) When you have too few columns to make an accurate prediction.", "B) As the number of features (dimensions) grows, the amount of data needed to generalize accurately grows exponentially, often degrading model performance.", "C) A bug in 3D plotting libraries.", "D) When arrays have mismatched shapes."],
            "correct_answer": "B"
        },
        {
            "question": "In A/B testing, what is 'Statistical Power'?",
            "options": ["A) The speed of the server.", "B) The probability that the test correctly rejects the null hypothesis when a specific alternative hypothesis is true (i.e., successfully detecting a real effect).", "C) The sample size required.", "D) The p-value multiplied by 100."],
            "correct_answer": "B"
        },
        {
            "question": "Which command removes all leading and trailing whitespace from every string in a pandas Series?",
            "options": ["A) df['Column'].strip()", "B) df['Column'].str.strip()", "C) df['Column'].trim()", "D) df['Column'].replace(' ', '')"],
            "correct_answer": "B"
        },
        # --- BATCH 33: Interactive Dashboards & Plotly Dash ---
        {
            "question": "What is the primary function of the `app.callback` decorator in a Plotly Dash application?",
            "options": ["A) To authenticate users logging into the dashboard.", "B) To link the interactive inputs (like dropdowns) to the outputs (like graphs) so they update dynamically.", "C) To download data from a SQL database.", "D) To style the HTML components with CSS."],
            "correct_answer": "B"
        },
        {
            "question": "In a Dash callback function, what do `Input` and `Output` objects correspond to?",
            "options": ["A) S3 buckets and EC2 instances.", "B) Machine learning training and testing sets.", "C) The `id` properties of specific HTML or Core Components defined in the app layout.", "D) Keyboard and mouse clicks."],
            "correct_answer": "C"
        },
        {
            "question": "Which module in Dash provides pre-built interactive components like sliders, dropdowns, and date pickers?",
            "options": ["A) dash.html", "B) dash.dcc (Dash Core Components)", "C) dash.dependencies", "D) dash.interactive"],
            "correct_answer": "B"
        },
        {
            "question": "What happens if a Dash callback function does not return the exact number of outputs specified in its decorator?",
            "options": ["A) It returns a list of Nones.", "B) It throws a CallbackException and the dashboard fails to update.", "C) It ignores the missing outputs.", "D) It defaults to zero."],
            "correct_answer": "B"
        },
        {
            "question": "How do you deploy a Dash app locally so you can view it in your web browser?",
            "options": ["A) dash.run()", "B) app.run_server(debug=True)", "C) server.start()", "D) app.deploy()"],
            "correct_answer": "B"
        },

        # --- BATCH 34: Algorithmic Efficiency & Big O Notation ---
        {
            "question": "When assessing the performance of a Python script, what does Big O notation (e.g., O(n)) measure?",
            "options": ["A) The exact number of seconds the script takes to run.", "B) The amount of RAM consumed by the script.", "C) How the runtime or space requirements scale as the size of the input data increases.", "D) The number of lines of code."],
            "correct_answer": "C"
        },
        {
            "question": "Searching for a specific value in an unsorted Python list has a worst-case time complexity of what?",
            "options": ["A) O(1)", "B) O(log n)", "C) O(n)", "D) O(n^2)"],
            "correct_answer": "C"
        },
        {
            "question": "Why is looking up a value in a Python dictionary (hash map) generally much faster than a list?",
            "options": ["A) Dictionaries are written in C++.", "B) Dictionaries have an average time complexity of O(1) for lookups due to hashing.", "C) Dictionaries automatically sort their data.", "D) Lists can only hold numbers."],
            "correct_answer": "B"
        },
        {
            "question": "If you write a nested `for` loop (a loop inside a loop) to iterate over a dataset of size N, what is the time complexity?",
            "options": ["A) O(N)", "B) O(N log N)", "C) O(N^2)", "D) O(1)"],
            "correct_answer": "C"
        },
        {
            "question": "Which sorting algorithm is typically implemented under the hood in Python's built-in `sort()` method (Timsort), and what is its average time complexity?",
            "options": ["A) Bubble Sort, O(N^2)", "B) Merge/Insertion Sort Hybrid, O(N log N)", "C) Quick Sort, O(N log N)", "D) Selection Sort, O(N^2)"],
            "correct_answer": "B"
        },

        # --- BATCH 35: Advanced Machine Learning & Ensembles ---
        {
            "question": "Which algorithm relies on building a series of decision trees sequentially, where each new tree tries to correct the errors of the previous ones?",
            "options": ["A) Random Forest", "B) K-Means", "C) Gradient Boosting (e.g., XGBoost)", "D) Naive Bayes"],
            "correct_answer": "C"
        },
        {
            "question": "What is the primary advantage of XGBoost over a standard Gradient Boosting Classifier in scikit-learn?",
            "options": ["A) XGBoost is completely unsupervised.", "B) XGBoost utilizes advanced regularization (L1/L2) and parallel processing, making it highly accurate and exceptionally fast.", "C) XGBoost only works on image data.", "D) XGBoost requires no hyperparameter tuning."],
            "correct_answer": "B"
        },
        {
            "question": "In an imbalanced business dataset (e.g., 99% legitimate transactions, 1% fraud), why is 'Accuracy' a misleading evaluation metric?",
            "options": ["A) Because accuracy cannot be calculated on binary data.", "B) Because a model predicting 'legitimate' every single time will still be 99% accurate while failing to catch any fraud.", "C) Because the algorithm will crash.", "D) Because accuracy requires a normal distribution."],
            "correct_answer": "B"
        },
        {
            "question": "To handle the imbalanced fraud dataset above, which technique allows you to artificially increase the number of minority class samples?",
            "options": ["A) Undersampling", "B) SMOTE (Synthetic Minority Over-sampling Technique)", "C) Principal Component Analysis", "D) K-Fold Cross Validation"],
            "correct_answer": "B"
        },
        {
            "question": "What is the difference between Bagging and Boosting in ensemble learning?",
            "options": ["A) They are identical concepts.", "B) Bagging trains models sequentially to reduce bias; Boosting trains them independently in parallel to reduce variance.", "C) Bagging trains models independently in parallel to reduce variance; Boosting trains them sequentially to reduce bias.", "D) Bagging is for regression, Boosting is for classification."],
            "correct_answer": "C"
        },

        # --- BATCH 36: Geospatial Analytics (GeoPandas) ---
        {
            "question": "Which Python library extends pandas to allow for spatial operations on geometric data types (like Points and Polygons)?",
            "options": ["A) NumPy", "B) Matplotlib", "C) GeoPandas", "D) Scikit-Learn"],
            "correct_answer": "C"
        },
        {
            "question": "In GeoPandas, what is a `geometry` column?",
            "options": ["A) A column containing the mathematical equations of the data.", "B) A special column holding Shapely objects (Points, Lines, Polygons) that defines the spatial location of the row.", "C) A column with strings representing city names.", "D) A 3D array."],
            "correct_answer": "B"
        },
        {
            "question": "If you want to find all customers (Points) located within a specific delivery zone (Polygon), which spatial join operation do you use?",
            "options": ["A) gpd.sjoin(customers, zones, op='within')", "B) pd.merge(customers, zones)", "C) customers.intersects(zones)", "D) gpd.concat([customers, zones])"],
            "correct_answer": "A"
        },
        {
            "question": "Which library is frequently paired with GeoPandas to create interactive, leaflet-based web maps in a Jupyter Notebook?",
            "options": ["A) Seaborn", "B) Plotly", "C) Folium", "D) PySpark"],
            "correct_answer": "C"
        },
        {
            "question": "What is a Coordinate Reference System (CRS) in geospatial analytics?",
            "options": ["A) The firewall protecting the map API.", "B) A framework used to precisely measure locations on the surface of the Earth as coordinates (e.g., EPSG:4326 for GPS).", "C) The Python library used to draw lines.", "D) A ranking system for map accuracy."],
            "correct_answer": "B"
        },

        # --- BATCH 37: Cloud Data Warehousing & BigQuery ---
        {
            "question": "When pulling data from Google BigQuery into a pandas DataFrame, which library is officially recommended for the fastest, most optimized extraction?",
            "options": ["A) psycopg2", "B) sqlite3", "C) google-cloud-bigquery (often used with pandas-gbq)", "D) sqlalchemy"],
            "correct_answer": "C"
        },
        {
            "question": "BigQuery is a 'columnar' database. What is a major advantage of columnar databases for business analytics compared to traditional row-based SQL?",
            "options": ["A) They are cheaper to install.", "B) They allow for vastly faster aggregation queries (like SUM or AVG) because they only read the specific columns requested, not the entire row.", "C) They support foreign keys automatically.", "D) They prevent data duplication completely."],
            "correct_answer": "B"
        },
        {
            "question": "If you execute a massive query in BigQuery via Python that processes 1 Terabyte of data, what happens?",
            "options": ["A) Your local Python RAM crashes.", "B) The query runs on Google's distributed servers, but you are billed based on the amount of data processed.", "C) The query is blocked by default.", "D) Python compresses the data locally before querying."],
            "correct_answer": "B"
        },
        {
            "question": "In cloud architectures, what does ETL stand for?",
            "options": ["A) Extract, Transform, Load", "B) Evaluate, Test, Learn", "C) External Transaction Ledger", "D) Event Time Logging"],
            "correct_answer": "A"
        },
        {
            "question": "Which tool is often used alongside Python to orchestrate and schedule complex ETL pipelines in the cloud?",
            "options": ["A) Apache Airflow", "B) Beautiful Soup", "C) TensorFlow", "D) Jupyter"],
            "correct_answer": "A"
        },

        # --- BATCH 38: Ethics, Privacy, and GDPR in Analytics ---
        {
            "question": "Under the GDPR, what does the principle of 'Data Minimization' mean for a data analyst?",
            "options": ["A) You must compress your CSV files to save server space.", "B) You should only collect and process the personal data that is strictly necessary for the specified purpose.", "C) You must delete all data after 30 days.", "D) You should only use small samples for machine learning."],
            "correct_answer": "B"
        },
        {
            "question": "If you are training a customer profiling model, what does 'Pseudonymization' involve?",
            "options": ["A) Deleting all customer records completely.", "B) Replacing direct identifiers (like names and emails) with artificial identifiers or hashes to reduce privacy risks.", "C) Making up fake customer data to train the model.", "D) Encrypting the entire hard drive."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Algorithmic Bias'?",
            "options": ["A) When an algorithm runs faster on a Mac than a PC.", "B) Systematic and repeatable errors in a computer system that create unfair outcomes, often due to prejudiced training data.", "C) A setting in scikit-learn to adjust learning rates.", "D) When a model prefers linear regression over decision trees."],
            "correct_answer": "B"
        },
        {
            "question": "If a user invokes their 'Right to be Forgotten' under GDPR, what must the analytics team typically do?",
            "options": ["A) Send the user a discount code to win them back.", "B) Ensure all personally identifiable information relating to that user is permanently erased from active databases and models.", "C) Move their data to an offshore server.", "D) Ignore it if the data is already in a pandas DataFrame."],
            "correct_answer": "B"
        },
        {
            "question": "In Python, which built-in module can be used to generate secure hashes (e.g., SHA-256) to anonymize email addresses before analysis?",
            "options": ["A) math", "B) hashlib", "C) random", "D) collections"],
            "correct_answer": "B"
        },

        # --- BATCH 39: Advanced Recommendation Metrics ---
        {
            "question": "When evaluating a Recommender System, what does 'Precision@K' measure?",
            "options": ["A) The percentage of the top K recommended items that are actually relevant to the user.", "B) The total number of items recommended.", "C) The exact rating the user will give to the Kth item.", "D) The speed of the recommendation generation."],
            "correct_answer": "A"
        },
        {
            "question": "How does 'Recall@K' differ from 'Precision@K'?",
            "options": ["A) Recall is always higher.", "B) Recall measures the percentage of all relevant items that were successfully retrieved in the top K recommendations.", "C) Recall only measures negative feedback.", "D) There is no difference."],
            "correct_answer": "B"
        },
        {
            "question": "What is the 'Jaccard Similarity' coefficient commonly used for in recommendation engines?",
            "options": ["A) Measuring the size of the database.", "B) Measuring similarity between finite sample sets (e.g., comparing the overlap of items purchased by two different users).", "C) Calculating the physical distance between servers.", "D) Determining the cost of the recommendation."],
            "correct_answer": "B"
        },
        {
            "question": "Which algorithm relies on factorizing a large user-item interaction matrix into lower-dimensional user and item latent factor matrices?",
            "options": ["A) Matrix Factorization (e.g., SVD)", "B) Decision Trees", "C) K-Means Clustering", "D) Logistic Regression"],
            "correct_answer": "A"
        },
        {
            "question": "In recommendation systems, what is a 'Content-Based' filtering approach?",
            "options": ["A) Recommending items based on what similar users liked.", "B) Recommending items based on the attributes or metadata of the items themselves (e.g., recommending an action movie because the user watched other action movies).", "C) Recommending the most popular items to everyone.", "D) Recommending items randomly."],
            "correct_answer": "B"
        },

        # --- BATCH 40: Advanced Deployment & FastAPI ---
        {
            "question": "FastAPI has largely replaced Flask for modern machine learning deployments. What is its primary advantage?",
            "options": ["A) It is older and more stable.", "B) It is significantly faster due to asynchronous programming (ASGI) and automatic data validation using Pydantic.", "C) It requires no coding.", "D) It only runs on Windows."],
            "correct_answer": "B"
        },
        {
            "question": "In FastAPI, what does the `@app.get(\"/predict\")` decorator do?",
            "options": ["A) It downloads the machine learning model.", "B) It defines a route, telling the API to trigger the associated function when a user sends an HTTP GET request to the '/predict' endpoint.", "C) It securely stores passwords.", "D) It prints the output to the console."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Pydantic' used for within a FastAPI application?",
            "options": ["A) Connecting to SQL databases.", "B) Defining data models and enforcing strict type checking/validation on incoming API requests.", "C) Creating interactive charts.", "D) Encrypting the API responses."],
            "correct_answer": "B"
        },
        {
            "question": "If your machine learning model takes 5 seconds to process an image, what Python feature allows your API to handle other requests while waiting for the model to finish?",
            "options": ["A) The `global` keyword", "B) Asynchronous programming (`async def` and `await`)", "C) Dictionary comprehensions", "D) `try / except` blocks"],
            "correct_answer": "B"
        },
        {
            "question": "What command launches a local server to test a FastAPI application named `main.py`?",
            "options": ["A) python main.py", "B) uvicorn main:app --reload", "C) fastapi start", "D) server.run()"],
            "correct_answer": "B"
        },

        # --- BATCH 41: Advanced Python Debugging & Testing ---
        {
            "question": "What does the built-in `pdb` module do in Python?",
            "options": ["A) Connects to a Postgres Database.", "B) Acts as an interactive source code debugger, allowing you to pause execution, inspect variables, and step through code line-by-line.", "C) Plots data distributions.", "D) Parses JSON files."],
            "correct_answer": "B"
        },
        {
            "question": "In modern Python, how do you insert a hardcoded breakpoint into your script to trigger the debugger instantly?",
            "options": ["A) stop()", "B) breakpoint()", "C) pause()", "D) debug_mode=True"],
            "correct_answer": "B"
        },
        {
            "question": "What is the purpose of the `pytest` framework in a data science project?",
            "options": ["A) To write unit tests that automatically verify your functions and data processing logic are working correctly.", "B) To test the internet speed.", "C) To scrape questions from testing websites.", "D) To automatically train machine learning models."],
            "correct_answer": "A"
        },
        {
            "question": "If you write `assert len(df) > 0` in your code, what happens if the DataFrame is empty?",
            "options": ["A) The code ignores it and moves on.", "B) The program halts and raises an AssertionError, preventing further execution on bad data.", "C) It automatically fills the DataFrame with zeros.", "D) It prints a warning but continues."],
            "correct_answer": "B"
        },
        {
            "question": "What does 'Mocking' mean in the context of unit testing an API connection script?",
            "options": ["A) Writing rude comments in the code.", "B) Creating fake, simulated responses for external dependencies (like an AWS bucket) so you can test your logic without making real network calls.", "C) Using dummy variables in a regression model.", "D) Stealing competitor data."],
            "correct_answer": "B"
        },
        
        # --- BATCH 42: Extra MSc Business Analytics Core Theory ---
        {
            "question": "In supply chain analytics, what does 'Safety Stock' refer to?",
            "options": ["A) Inventory held to protect against uncertainties in demand or supply lead time.", "B) Stock that is physically locked in a safe.", "C) Inventory that has passed expiration.", "D) The maximum capacity of a warehouse."],
            "correct_answer": "A"
        },
        {
            "question": "When building an attribution model in marketing analytics, what does a 'Last-Click' model assume?",
            "options": ["A) All marketing channels deserve equal credit for the sale.", "B) The very last touchpoint a customer clicked before buying deserves 100% of the credit for the conversion.", "C) The first ad they saw gets 100% of the credit.", "D) Only social media ads get credit."],
            "correct_answer": "B"
        },
        {
            "question": "What is the primary formula for calculating Customer Lifetime Value (CLV)?",
            "options": ["A) Average Order Value / Profit Margin", "B) (Average Purchase Value * Average Purchase Frequency) * Customer Lifespan", "C) Total Revenue - Total Costs", "D) Customer Acquisition Cost * Lifespan"],
            "correct_answer": "B"
        },
        {
            "question": "In a churn prediction model, if you optimize your model to have extremely high Recall at the expense of Precision, what is the business consequence?",
            "options": ["A) You will miss many customers who are actually churning.", "B) You will catch almost all churning customers, but you will also falsely predict many loyal customers are churning (False Positives), potentially wasting retention budgets.", "C) You will have zero False Positives.", "D) The model will refuse to predict."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Cohort Analysis' in user retention analytics?",
            "options": ["A) Grouping users by their astrological sign.", "B) Tracking the behavior and retention of a specific group of users who share a common characteristic (like the month they signed up) over time.", "C) Analyzing data using only the columns that correlate with revenue.", "D) Running a random forest model multiple times."],
            "correct_answer": "B"
        },
        # --- BATCH 43: Advanced Pandas Edge-Cases ---
        {
            "question": "You have two DataFrames with timestamps that don't match exactly, and you want to join them based on the closest preceding time. Which pandas function is designed specifically for this?",
            "options": ["A) pd.merge_time()", "B) pd.merge_asof()", "C) pd.concat()", "D) pd.join_nearest()"],
            "correct_answer": "B"
        },
        {
            "question": "Which pandas method allows you to evaluate a string describing operations on DataFrame columns, often executing much faster on large datasets than standard Python syntax?",
            "options": ["A) df.execute()", "B) df.run()", "C) df.eval()", "D) df.calculate()"],
            "correct_answer": "C"
        },
        {
            "question": "How do you select rows from a DataFrame using a boolean expression in a string format (e.g., \"Sales > 1000 and Region == 'North'\")?",
            "options": ["A) df.filter()", "B) df.find()", "C) df.search()", "D) df.query()"],
            "correct_answer": "D"
        },
        {
            "question": "What is the primary benefit of converting a column with a small number of unique string values (like 'State' or 'Country') into the pandas `category` data type?",
            "options": ["A) It automatically one-hot encodes the data.", "B) It drastically reduces memory usage and speeds up operations like sorting and grouping.", "C) It translates the text into English.", "D) It prevents missing values."],
            "correct_answer": "B"
        },
        {
            "question": "If you have a MultiIndex DataFrame, how do you slice data based on the second level of the index without specifying the first level?",
            "options": ["A) Using pd.IndexSlice", "B) Using df.loc[:, 'Level2']", "C) Using df.iloc[1]", "D) You cannot skip the first level."],
            "correct_answer": "A"
        },
        {
            "question": "What does the `df.clip(lower=0, upper=100)` method do?",
            "options": ["A) Deletes rows outside the 0-100 range.", "B) Forces all values below 0 to become 0, and all values above 100 to become 100.", "C) Normalizes the data between 0 and 100.", "D) Rounds the numbers."],
            "correct_answer": "B"
        },
        {
            "question": "Which pandas method instantly creates a correlation matrix for all numerical columns in a DataFrame?",
            "options": ["A) df.correlate()", "B) df.matrix()", "C) df.corr()", "D) df.stats()"],
            "correct_answer": "C"
        },
        {
            "question": "If `df.duplicated()` returns a boolean mask of duplicate rows, which argument ensures that ALL duplicates are marked True, not just the subsequent ones?",
            "options": ["A) keep='all'", "B) keep=False", "C) mark_all=True", "D) first=False"],
            "correct_answer": "B"
        },
        {
            "question": "What does the `df.memory_usage(deep=True)` command provide?",
            "options": ["A) The CPU usage of the DataFrame.", "B) The precise RAM consumption of each column, including the memory used by strings/objects.", "C) A graph of memory over time.", "D) It clears the memory cache."],
            "correct_answer": "B"
        },
        {
            "question": "You have a column 'Tags' containing lists of strings. Which method transforms each element in those lists into a separate row?",
            "options": ["A) df.expand()", "B) df.split()", "C) df.explode()", "D) df.unstack()"],
            "correct_answer": "C"
        },

        # --- BATCH 44: Deep Learning Architectures (CNNs & RNNs) ---
        {
            "question": "In deep learning, what is a Convolutional Neural Network (CNN) primarily designed to analyze?",
            "options": ["A) Text data", "B) Time-series data", "C) Grid-like topologies, such as Image data", "D) Audio files"],
            "correct_answer": "C"
        },
        {
            "question": "Which type of neural network architecture has 'memory' and is specifically designed to handle sequential data, like text or time-series?",
            "options": ["A) Recurrent Neural Network (RNN)", "B) Convolutional Neural Network (CNN)", "C) Multilayer Perceptron (MLP)", "D) Autoencoder"],
            "correct_answer": "A"
        },
        {
            "question": "What problem does an LSTM (Long Short-Term Memory) network solve in traditional RNNs?",
            "options": ["A) The inability to process images.", "B) The Vanishing Gradient Problem, allowing the network to remember long-term dependencies in sequential data.", "C) The slow training speed.", "D) The lack of an activation function."],
            "correct_answer": "B"
        },
        {
            "question": "In the context of training a neural network, what is 'Backpropagation'?",
            "options": ["A) Moving data backwards through a pipeline.", "B) The algorithm used to calculate the gradient of the loss function with respect to the network's weights, allowing the model to learn and update.", "C) Reversing an array.", "D) A type of dropout layer."],
            "correct_answer": "B"
        },
        {
            "question": "What happens if your Learning Rate is set too high during neural network training?",
            "options": ["A) The model learns perfectly in one epoch.", "B) The model may converge too quickly to a suboptimal solution or drastically overshoot the global minimum, causing loss to diverge.", "C) The model will run out of memory.", "D) The model will train infinitely slow."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Transfer Learning'?",
            "options": ["A) Transferring data via FTP.", "B) Taking a pre-trained model (trained on a massive dataset) and fine-tuning it on a smaller, specific dataset for a similar task.", "C) Moving code from Python to C++.", "D) Converting a classification model to a regression model."],
            "correct_answer": "B"
        },
        {
            "question": "Which Keras layer is used to convert multi-dimensional output (like from a convolutional layer) into a 1D array before passing it to a Dense layer?",
            "options": ["A) DenseLayer", "B) Flatten", "C) MaxPool2D", "D) Dropout"],
            "correct_answer": "B"
        },
        {
            "question": "In Natural Language Processing with deep learning, what is an 'Embedding' layer?",
            "options": ["A) A layer that inserts images into text.", "B) A layer that translates words into dense vectors of fixed size, capturing semantic meaning and relationships between words.", "C) A layer that removes punctuation.", "D) A layer that counts word frequency."],
            "correct_answer": "B"
        },
        {
            "question": "What is the purpose of an 'Autoencoder' neural network?",
            "options": ["A) To automatically encode Python code.", "B) To learn a compressed, latent representation of input data (dimensionality reduction or anomaly detection) by attempting to reconstruct the input at the output.", "C) To scrape websites.", "D) To predict future stock prices."],
            "correct_answer": "B"
        },
        {
            "question": "Which loss function is standard for training a neural network on a binary classification problem (e.g., Churn vs. No Churn)?",
            "options": ["A) Mean Squared Error", "B) Categorical Crossentropy", "C) Binary Crossentropy", "D) Mean Absolute Error"],
            "correct_answer": "C"
        },

        # --- BATCH 45: LLMs, Generative AI & RAG ---
        {
            "question": "What does the architecture 'Transformer' (the 'T' in GPT) rely on heavily to understand the context of words in a sentence?",
            "options": ["A) Convolutions", "B) Recurrence", "C) The Attention Mechanism (Self-Attention)", "D) Random Forests"],
            "correct_answer": "C"
        },
        {
            "question": "In the context of integrating LLMs into business applications, what does RAG stand for?",
            "options": ["A) Random Access Generation", "B) Retrieval-Augmented Generation", "C) Regional Analytics Group", "D) Relational Algorithm Grid"],
            "correct_answer": "B"
        },
        {
            "question": "What is the primary benefit of a RAG architecture when querying a Large Language Model?",
            "options": ["A) It makes the model train faster.", "B) It retrieves relevant facts from an external business database and feeds them to the LLM, vastly reducing hallucinations and allowing the LLM to answer using private company data.", "C) It translates code automatically.", "D) It compresses the model size."],
            "correct_answer": "B"
        },
        {
            "question": "When working with OpenAI's API or similar LLMs in Python, what is a 'Token'?",
            "options": ["A) A password used for login.", "B) The basic unit of text processed by the model (often a word or part of a word).", "C) A database query.", "D) A Python dictionary key."],
            "correct_answer": "B"
        },
        {
            "question": "What is the role of a 'Vector Database' (like Pinecone or ChromaDB) in modern AI applications?",
            "options": ["A) To store SQL tables.", "B) To store and quickly search through mathematical vector embeddings of text, images, or audio.", "C) To render SVG graphics.", "D) To host websites."],
            "correct_answer": "B"
        },
        {
            "question": "What does 'Temperature' control when generating text with an LLM via an API?",
            "options": ["A) The physical heat of the GPU.", "B) The randomness or creativity of the output (lower means more deterministic, higher means more random/creative).", "C) The length of the response.", "D) The speed of the generation."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Prompt Engineering'?",
            "options": ["A) Building bridges with code.", "B) The process of structuring text input to an LLM to effectively guide it toward generating the desired output.", "C) Writing SQL queries.", "D) Fixing hardware bugs."],
            "correct_answer": "B"
        },
        {
            "question": "When fine-tuning an LLM for a specific business task, what does 'Few-Shot Prompting' refer to?",
            "options": ["A) Only allowing the model to try three times.", "B) Providing the model with a few examples of inputs and desired outputs within the prompt itself to teach it the format.", "C) Using only a small amount of RAM.", "D) Deleting data."],
            "correct_answer": "B"
        },
        {
            "question": "Which Python library framework is widely used to build applications powered by LLMs, allowing you to chain together prompts, models, and agents?",
            "options": ["A) Matplotlib", "B) LangChain", "C) PySpark", "D) BeautifulSoup"],
            "correct_answer": "B"
        },
        {
            "question": "What is a 'System Prompt' in the context of conversational AI?",
            "options": ["A) An error message from Windows.", "B) A high-level instruction given to the model that sets its behavior, persona, and boundaries for the entire conversation.", "C) The user's typed question.", "D) The API key."],
            "correct_answer": "B"
        },

        # --- BATCH 46: Causal Inference & Advanced Experimentation ---
        {
            "question": "In business analytics, why does 'Correlation not imply Causation'?",
            "options": ["A) Because math is flawed.", "B) Because two variables moving together might both be influenced by an unseen third variable (a confounder).", "C) Because p-values are always wrong.", "D) Because regression models cannot handle big data."],
            "correct_answer": "B"
        },
        {
            "question": "Which advanced statistical method compares the changes in outcomes over time between a treatment group and a control group to estimate causal effects?",
            "options": ["A) K-Means Clustering", "B) Difference-in-Differences (DiD)", "C) Naive Bayes", "D) Principal Component Analysis"],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Propensity Score Matching' used for in observational data analysis?",
            "options": ["A) Matching customers on Tinder.", "B) Attempting to estimate the effect of a treatment by pairing treated and untreated subjects with similar characteristics, simulating a randomized control trial.", "C) Calculating credit scores.", "D) Scoring exam papers."],
            "correct_answer": "B"
        },
        {
            "question": "In marketing, what does 'Uplift Modeling' seek to predict?",
            "options": ["A) The total revenue of the company.", "B) The incremental impact of a treatment (like sending an ad) on an individual's behavior, identifying those who will buy *only if* targeted.", "C) The physical weight of a product.", "D) The stock price."],
            "correct_answer": "B"
        },
        {
            "question": "What is a 'Confounding Variable'?",
            "options": ["A) A variable that crashes your Python script.", "B) An extraneous variable that influences both the dependent and independent variables, causing a spurious association.", "C) A variable with missing values.", "D) A string variable in a numeric column."],
            "correct_answer": "B"
        },
        {
            "question": "Which library in Python is specifically built for applying causal inference methods to business datasets?",
            "options": ["A) Seaborn", "B) DoWhy (or EconML)", "C) Beautiful Soup", "D) SQLAlchemy"],
            "correct_answer": "B"
        },
        {
            "question": "What is an 'A/A Test'?",
            "options": ["A) A test with 100% accuracy.", "B) A sanity check where two identical versions of a page are tested against each other to ensure the testing platform is measuring variations correctly (no statistical difference should be found).", "C) A test for Apple products.", "D) A double A/B test."],
            "correct_answer": "B"
        },
        {
            "question": "If your A/B test results are statistically significant, but the increase in revenue is only £0.01 per user, what does this test lack?",
            "options": ["A) Statistical Significance", "B) Practical (Business) Significance", "C) Data", "D) Python code"],
            "correct_answer": "B"
        },
        {
            "question": "What is the 'Network Effect' (or interference) problem in A/B testing on social platforms?",
            "options": ["A) The Wi-Fi goes down.", "B) When the treatment applied to one user affects the behavior of another user in the control group (e.g., matching algorithms), violating the assumption of independence.", "C) When code is leaked online.", "D) When servers overheat."],
            "correct_answer": "B"
        },
        {
            "question": "In hypothesis testing, what is the 'p-value hacking' (p-hacking) fallacy?",
            "options": ["A) Illegally accessing a competitor's database.", "B) Manipulating data analysis or exhaustively testing different variables until a statistically significant pattern is found by chance, rather than formulating a hypothesis first.", "C) Forgetting to import pandas.", "D) Hacking into a Jupyter Notebook."],
            "correct_answer": "B"
        },

        # --- BATCH 47: Software Engineering for Data (OOP & Clean Code) ---
        {
            "question": "In Python Object-Oriented Programming, what does the `self` keyword represent inside a class method?",
            "options": ["A) A global variable.", "B) The instance of the class itself, allowing access to its attributes and methods.", "C) The parent class.", "D) The Python interpreter."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Inheritance' in OOP?",
            "options": ["A) Receiving money.", "B) A mechanism where a new class derives properties and behaviors (methods) from an existing parent class.", "C) Copying and pasting code.", "D) Importing a module."],
            "correct_answer": "B"
        },
        {
            "question": "Which Python decorator allows you to define a method inside a class that can be called directly on the class itself, without needing to create an object/instance first?",
            "options": ["A) @staticmethod or @classmethod", "B) @property", "C) @abstract", "D) @override"],
            "correct_answer": "A"
        },
        {
            "question": "What are 'Dunder' methods in Python (e.g., `__len__` or `__str__`)?",
            "options": ["A) Methods that are broken.", "B) 'Double Underscore' special or magic methods that allow you to define how your custom objects interact with built-in Python operations like `len()` or `print()`.", "C) Methods used for hacking.", "D) Methods that delete data."],
            "correct_answer": "B"
        },
        {
            "question": "What is 'Type Hinting' in modern Python (e.g., `def calculate(x: int) -> float:`)?",
            "options": ["A) It forces Python to crash if the wrong type is passed.", "B) It provides a visual guide for developers and IDEs about what data types a function expects and returns, improving code readability and debugging.", "C) It converts integers to floats automatically.", "D) It makes the code run twice as fast."],
            "correct_answer": "B"
        },
        {
            "question": "What is a 'Docstring'?",
            "options": ["A) A string containing medical data.", "B) A multi-line string literal (using `\"\"\"`) placed immediately after a function or class definition to document what it does.", "C) A method to join strings.", "D) A string formatted as JSON."],
            "correct_answer": "B"
        },
        {
            "question": "In software engineering for analytics, what does 'CI/CD' stand for?",
            "options": ["A) Continuous Integration / Continuous Deployment", "B) Centralized Information / Code Distribution", "C) Custom Implementation / Client Delivery", "D) Code Input / Code Debugging"],
            "correct_answer": "A"
        },
        {
            "question": "Why is it best practice to use a `requirements.txt` file or a `Pipfile` in a Python project?",
            "options": ["A) To list the business requirements of the project.", "B) To specify the exact external libraries and their versions needed to run the code, ensuring reproducibility on other machines.", "C) To write passwords.", "D) To store raw data."],
            "correct_answer": "B"
        },
        {
            "question": "What is the PEP 8 standard?",
            "options": ["A) A data privacy law.", "B) The official style guide for writing clean, readable Python code (covering things like indentation, naming conventions, and line length).", "C) A machine learning algorithm.", "D) A SQL database engine."],
            "correct_answer": "B"
        },
        {
            "question": "What does 'Refactoring' mean in programming?",
            "options": ["A) Changing the business objective of the code.", "B) Restructuring existing code without changing its external behavior, to improve readability, reduce complexity, or make it more efficient.", "C) Deleting the repository.", "D) Writing code for a factory."],
            "correct_answer": "B"
        },

        # --- BATCH 48: The Master's Capstone (Mixed Advanced Analytics) ---
        {
            "question": "In Survival Analysis (often used for estimating customer lifespan), what is a 'Kaplan-Meier Estimator'?",
            "options": ["A) A neural network.", "B) A non-parametric statistic used to estimate the survival function from lifetime data, tracking the probability of 'surviving' past a certain time.", "C) A metric for server health.", "D) A clustering algorithm."],
            "correct_answer": "B"
        },
        {
            "question": "What does 'Right-Censored Data' mean when analyzing customer churn?",
            "options": ["A) Data from right-handed customers.", "B) The study ended or we analyzed the data before the customer churned, meaning we know they survived up to a certain point, but not their ultimate lifespan.", "C) Data that has been deleted due to privacy laws.", "D) Data that is perfectly accurate."],
            "correct_answer": "B"
        },
        {
            "question": "In pricing analytics, what does Price Elasticity of Demand measure?",
            "options": ["A) How stretchy the product packaging is.", "B) The percentage change in quantity demanded resulting from a 1% change in price.", "C) The total profit of a product.", "D) The competitor's price."],
            "correct_answer": "B"
        },
        {
            "question": "If a product has a Price Elasticity of -2.5, it is considered:",
            "options": ["A) Inelastic (demand hardly changes with price).", "B) Elastic (demand is highly sensitive to price changes).", "C) Perfectly rigid.", "D) Unprofitable."],
            "correct_answer": "B"
        },
        {
            "question": "What is the 'Silhouette Score' used for in unsupervised learning?",
            "options": ["A) To measure the accuracy of a decision tree.", "B) To evaluate the quality of clusters in K-Means by calculating how similar an object is to its own cluster compared to other clusters.", "C) To determine image brightness.", "D) To predict text sequences."],
            "correct_answer": "B"
        },
        {
            "question": "In a logistics context, what is the 'Traveling Salesperson Problem (TSP)'?",
            "options": ["A) Hiring sales staff.", "B) An optimization problem finding the shortest possible route that visits every node exactly once and returns to the origin city.", "C) Tracking employee expenses.", "D) Forecasting sales during a holiday."],
            "correct_answer": "B"
        },
        {
            "question": "What does the `df.xs()` method do in pandas?",
            "options": ["A) Extends the DataFrame.", "B) Returns a cross-section from a Series/DataFrame, incredibly useful for extracting data at a specific level of a MultiIndex.", "C) Converts data to XML.", "D) Drops columns."],
            "correct_answer": "B"
        },
        {
            "question": "In an XGBoost model, what does the 'learning_rate' (or eta) parameter control?",
            "options": ["A) The speed of the computer fan.", "B) The step size shrinkage used in update to prevent overfitting. Lower values make the model more robust but require more trees to train.", "C) The amount of RAM allocated.", "D) The maximum depth of the trees."],
            "correct_answer": "B"
        },
        {
            "question": "What is the core philosophy behind Agile methodology in an analytics data team?",
            "options": ["A) Writing all code in one giant script.", "B) Iterative, incremental development allowing teams to deliver small, functional pieces of a project quickly and adapt to changing business requirements.", "C) Waiting a year to deliver a perfect model.", "D) Never documenting code."],
            "correct_answer": "B"
        },
        {
            "question": "Congratulations! You are taking a Python MCQ exam for an MSc in Business Analytics. If you master these 500 concepts, what is the most likely outcome?",
            "options": ["A) You will panic.", "B) You will confidently crush the exam, demonstrate master-level Python proficiency, and excel in your Business Analytics career.", "C) You will forget it all.", "D) You will switch to Excel."],
            "correct_answer": "B"
        }
    ]
    # Randomly select 15 questions from the bank for this session
    return random.sample(massive_question_bank, min(15, len(massive_question_bank)))

# --- 2. Session State Management ---
if 'current_exam' not in st.session_state:
    st.session_state.current_exam = fetch_dynamic_questions()
if 'is_submitted' not in st.session_state:
    st.session_state.is_submitted = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

exam_questions = st.session_state.current_exam

# --- 3. Main UI ---
st.title("Python Programming Mock Exam")
st.markdown("---")

# VIEW A: THE EXAM IS ONGOING
if not st.session_state.is_submitted:
    
    with st.form("exam_form"):
        st.write("Select the correct answers below and click submit.")
        
        user_selections = {}
        for i, q in enumerate(exam_questions):
            st.subheader(f"Question {i + 1}")
            st.write(q['question'])
            
            user_selections[i] = st.radio(
                "Choose your answer:", 
                q['options'], 
                key=f"q_{i}",
                index=None
            )
            st.markdown("---")
            
        submitted = st.form_submit_button("Submit Exam")
        
        if submitted:
            # Save answers, flip the flag to True, and instantly reload the page
            st.session_state.user_answers = user_selections
            st.session_state.is_submitted = True
            st.rerun()

# VIEW B: THE EXAM IS SUBMITTED (Inline Results)
else:
    score = 0
    total = len(exam_questions)
    
    # Calculate score
    for i, q in enumerate(exam_questions):
        ans = st.session_state.user_answers.get(i)
        if ans and ans[0] == q['correct_answer']:
            score += 1
            
    # Show the final score at the top
    percentage = (score / total) * 100
    if percentage >= 70:
        st.success(f"### Final Score: {score} / {total} ({percentage:.2f}%)")
    else:
        st.error(f"### Final Score: {score} / {total} ({percentage:.2f}%)")
    st.markdown("---")
    
    # Display the inline feedback
    for i, q in enumerate(exam_questions):
        st.subheader(f"Question {i + 1}")
        st.write(q['question'])
        
        user_choice = st.session_state.user_answers.get(i)
        user_letter = user_choice[0] if user_choice else None
        correct_letter = q['correct_answer']
        
        # Format the options with inline colors
        for option in q['options']:
            opt_letter = option[0]
            
            if opt_letter == correct_letter:
                st.markdown(f"<div style='color: #155724; background-color: #d4edda; padding: 8px; border-radius: 5px; margin-bottom: 5px;'>✅ <strong>{option}</strong></div>", unsafe_allow_html=True)
            elif opt_letter == user_letter and user_letter != correct_letter:
                st.markdown(f"<div style='color: #721c24; background-color: #f8d7da; padding: 8px; border-radius: 5px; text-decoration: line-through; margin-bottom: 5px;'>❌ {option}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: #555555; padding: 8px; margin-bottom: 5px;'>⚪ {option}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
    # Reset button to fetch 15 new random questions
    if st.button("Take a New Random Exam"):
        st.session_state.is_submitted = False
        st.session_state.user_answers = {}
        del st.session_state.current_exam
        st.rerun()