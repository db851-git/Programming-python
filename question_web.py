import streamlit as st

# --- Test Data ---
exam_questions = [
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
    }
]


# --- UI Layout ---
st.title("Python Programming Mock Exam")
st.write("Select the correct answers below and click submit at the bottom of the page.")
st.markdown("---")

# Create a form so the page doesn't reload after every click
with st.form("exam_form"):
    user_selections = {}
    
    # Generate the questions and radio buttons dynamically
    for i, q in enumerate(exam_questions):
        st.subheader(f"Question {i + 1}")
        st.write(q['question'])
        
        # Display options as a clickable radio button group
        user_selections[i] = st.radio(
            "Choose your answer:", 
            q['options'], 
            key=f"question_{i}",
            index=None # Ensures no default option is selected
        )
        st.markdown("---")
        
    # Submit button
    submitted = st.form_submit_button("Submit Exam")

# --- Scoring Logic (Runs only when submit is clicked) ---
if submitted:
    score = 0
    total_questions = len(exam_questions)
    
    st.header("Exam Results")
    
    for i, q in enumerate(exam_questions):
        user_choice = user_selections[i]
        
        # Check if the user missed a question
        if user_choice is None:
            st.warning(f"Question {i + 1}: You left this blank! The correct answer was {q['correct_answer']}.")
            continue
            
        # Extract just the first letter (A, B, C, or D) from the user's choice
        user_letter = user_choice[0]
        
        if user_letter == q['correct_answer']:
            score += 1
        else:
            st.error(f"Question {i + 1}: Incorrect. You chose {user_letter}, but the correct answer was {q['correct_answer']}.")
            
    # Final Score Display
    percentage = (score / total_questions) * 100
    if percentage >= 70:
        st.success(f"### Final Score: {score} / {total_questions} ({percentage:.2f}%) - Great job!")
    else:
        st.info(f"### Final Score: {score} / {total_questions} ({percentage:.2f}%) - Keep practicing!")