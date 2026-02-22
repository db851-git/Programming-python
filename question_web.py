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