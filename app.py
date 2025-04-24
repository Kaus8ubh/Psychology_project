import streamlit as st
import numpy as np
import random
import joblib
import pandas as pd
from models.classification_model import RandomForestPipeline

# App configuration
st.set_page_config(
    page_title="Attachment Style Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Questions from preprocessing module ---
authoritative_qs = [
    "My parents are responsive to my feelings and needs.",
    "My parents consider my wishes before asking me to do something.",
    "My parents explain how they feel about my good/bad behavior.",
    "My parents encourage me to talk about my feelings and problems.",
    "My parents encourage me to freely express my thoughts, even if I disagree with them.",
    "My parents explain the reasons behind their expectations.",
    "My parents provide comfort and understanding when I am upset.",
    "My parents compliment me.",
    "My parents consider my preferences when making family plans.",
    "My parents respect my opinion and encourage me to express it.",
    "My parents treat me as an equal member of the family.",
    "My parents provide reasons for their expectations of me.",
    "My parents and I have warm and intimate times together."
]

authoritarian_qs = [
    "When I ask why I have to do something, my parents say it's because they said so.",
    "My parents punish me by taking away privileges.",
    "My parents yell when they disapprove of my behavior.",
    "My parents explode in anger towards me.",
    "My parents spank me when they don't like what I do or say.",
    "My parents use criticism to make me improve my behavior.",
    "My parents use threats as punishment with little justification.",
    "My parents punish me by withholding emotional expressions.",
    "My parents openly criticize me when my behavior doesn't meet expectations.",
    "My parents struggle to change how I think or feel about things.",
    "My parents point out my past behavioral problems to ensure I don't repeat them.",
    "My parents remind me that they are my parents.",
    "My parents remind me of all the things they do for me."
]

permissive_qs = [
    "My parents find it difficult to discipline me.",
    "My parents give in to me when I cause a commotion about something.",
    "My parents spoil me.",
    "My parents ignore my bad behavior."
]

# Define emojis for Likert scale
emoji_scale = ["😠", "🙁", "😐", "🙂", "😄"]

# --- Sidebar for app navigation and info ---
with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2017/01/31/19/26/avatar-2026744_1280.png", width=100)
    st.title("About this App")
    st.info("""
    This app predicts your attachment style based on your parenting experience. 
    
    **How it works:**
    1. Answer questions about your childhood experiences
    2. Our AI analyzes your responses
    3. Get insights about your attachment style
    
    **Your data is private** and not stored permanently.
    """)
    
    # Add progress tracker in sidebar
    if 'page' in st.session_state and 'all_qs' in st.session_state:
        progress = min(1.0, (st.session_state.page + 1) * 4 / len(st.session_state.all_qs))
        st.progress(progress)
        st.write(f"Progress: {int(progress * 100)}%")

# --- Initialize session state ---
if 'page' not in st.session_state:
    st.session_state.page = 0
    st.session_state.answers = {}  # Store answers as dictionary with question as key
    st.session_state.test_done = False
    st.session_state.show_explanation = False

# Combine all questions only once and shuffle
if 'all_qs' not in st.session_state:
    all_qs = authoritative_qs + authoritarian_qs + permissive_qs
    random.shuffle(all_qs)
    st.session_state.all_qs = all_qs

# --- App Title & Introduction ---
st.title("🧠 Attachment Style Predictor")

if not st.session_state.test_done:
    st.write("""
    ### Welcome to Your Attachment Style Journey!
    
    This assessment helps you understand how your childhood experiences may have shaped your attachment patterns.
    Answer each question based on your experiences growing up with your primary caregivers.
    """)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📋 Assessment", "ℹ️ About Attachment Styles"])
    
    with tab1:
        # --- Display the current set of questions ---
        st.subheader("Rate how much you agree with each statement:")
        
        start_index = st.session_state.page * 4
        end_index = min(start_index + 4, len(st.session_state.all_qs))
        page_questions = st.session_state.all_qs[start_index:end_index]
        
        for q in page_questions:
            # Get existing value if available
            default_val = st.session_state.answers.get(q, 3)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                score = st.slider(
                    q, 
                    1, 5, 
                    default_val,
                    help="1=Strongly Disagree, 5=Strongly Agree",
                    key=f"slider_{q}"
                )
            with col2:
                st.write(f"**{emoji_scale[score-1]}** {score}/5")
            
            # Store answer in session state
            st.session_state.answers[q] = score
        
        # --- Buttons for Pagination ---
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.page > 0:
                if st.button('⬅️ Previous'):
                    st.session_state.page -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.page < (len(st.session_state.all_qs) - 1) // 4:
                if st.button('Next ➡️'):
                    st.session_state.page += 1
                    st.rerun()
                    
        # Show the prediction button only on the last page or when all questions are answered
        if st.session_state.page >= (len(st.session_state.all_qs) - 1) // 4 or len(st.session_state.answers) == len(st.session_state.all_qs):
            with col2:
                if st.button("🔮 Predict My Style", type="primary"):
                    # Calculate average scores for each parenting style
                    answers_list = []
                    for q in st.session_state.all_qs:
                        if q in st.session_state.answers:
                            answers_list.append(st.session_state.answers[q])
                        else:
                            answers_list.append(3)  # Default value for unanswered questions
                    
                    # Calculate average scores for each parenting style
                    avg_authoritative = np.mean([score for score, q in zip(answers_list, st.session_state.all_qs) if q in authoritative_qs])
                    avg_authoritarian = np.mean([score for score, q in zip(answers_list, st.session_state.all_qs) if q in authoritarian_qs])
                    avg_permissive = np.mean([score for score, q in zip(answers_list, st.session_state.all_qs) if q in permissive_qs])
                    
                    # Store in session state for results page
                    st.session_state.avg_authoritative = avg_authoritative
                    st.session_state.avg_authoritarian = avg_authoritarian
                    st.session_state.avg_permissive = avg_permissive
                    
                    input_array = np.array([avg_authoritative, avg_authoritarian, avg_permissive])
                    
                    try:
                        # Load pipeline and trained model
                        rf_pipeline = RandomForestPipeline()
                        rf_pipeline.load_and_prepare_data("scores.csv")
                        model = joblib.load("random_forest_model.pkl")
                        rf_pipeline.model = model
                        
                        # Get parenting style with highest average score
                        parenting_styles = ['authoritative', 'authoritarian', 'permissive']
                        predicted_parenting_style = parenting_styles[np.argmax(input_array)]
                        st.session_state.parenting_style = predicted_parenting_style
                        
                        # Encode the parenting style
                        X_user = rf_pipeline.label_encoder_X.transform([predicted_parenting_style]).reshape(-1, 1)
                        
                        # Predict attachment style probabilities
                        pred_probs = model.predict_proba(X_user)[0]
                        attachment_styles = ['Secure', 'Avoidant', 'Anxious']
                        pred_percentages = [prob * 100 for prob in pred_probs]
                        
                        # Store results in session state
                        st.session_state.attachment_styles = attachment_styles
                        st.session_state.pred_percentages = pred_percentages
                        st.session_state.highest_style = attachment_styles[np.argmax(pred_percentages)]
                        st.session_state.highest_percentage = max(pred_percentages)
                        
                        st.session_state.test_done = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
    
    with tab2:
        st.subheader("Understanding Attachment Styles")
        
        st.write("""
        **Secure Attachment:** People with secure attachment find it easy to trust others and form close relationships. They have a positive view of themselves and others.
        
        **Avoidant Attachment:** People with avoidant attachment tend to avoid emotional closeness and may prioritize independence over relationships.
        
        **Anxious Attachment:** People with anxious attachment often worry about their relationships and seek high levels of closeness, approval, and responsiveness.
        
        These attachment patterns form in childhood based on our relationships with primary caregivers and can influence our adult relationships.
        """)

# --- Display Results ---
if st.session_state.test_done:
    st.balloons()
    
    # Create columns for results display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title(f"Your Attachment Style: {st.session_state.highest_style}")
        st.subheader(f"{st.session_state.highest_percentage:.1f}% match")
        
        # Create a horizontal bar chart for attachment style percentages
        chart_data = pd.DataFrame({
            'Style': st.session_state.attachment_styles,
            'Percentage': st.session_state.pred_percentages
        })
        
        st.bar_chart(chart_data.set_index('Style'), height=200)
        
        # Display parenting style information
        st.subheader("Parenting Style Analysis")
        st.write(f"Based on your responses, your predominant parenting experience was **{st.session_state.parenting_style}**.")
        
        # Create a radar chart for parenting styles
        parenting_data = {
            'Category': ['Authoritative', 'Authoritarian', 'Permissive'],
            'Score': [st.session_state.avg_authoritative, st.session_state.avg_authoritarian, st.session_state.avg_permissive]
        }
        
        parenting_df = pd.DataFrame(parenting_data)
        st.write("Your Parenting Style Scores:")
        st.bar_chart(parenting_df.set_index('Category'), height=200)
        
    with col2:
        st.image(f"https://cdn.pixabay.com/photo/2016/04/01/10/04/brain-1299981_1280.png", width=200)
        
        # Explanation based on attachment style
        if st.session_state.highest_style == "Secure":
            st.success("""
            **Secure Attachment**
            
            You tend to have healthy, trusting relationships. You're comfortable with intimacy and independence.
            """)
        elif st.session_state.highest_style == "Avoidant":
            st.info("""
            **Avoidant Attachment**
            
            You may value independence and self-sufficiency. You might find it uncomfortable to fully trust or depend on others.
            """)
        else:
            st.warning("""
            **Anxious Attachment**
            
            You might seek high levels of intimacy, approval, and responsiveness from partners. You may worry about your relationships.
            """)
    
    # Detailed explanation toggle
    with st.expander("See detailed explanation"):
        st.write("""
        ### How Your Childhood Affects Your Attachment
        
        Your early experiences with caregivers create internal working models about relationships that can persist into adulthood. These models influence how you:
        
        - Form and maintain relationships
        - Respond to emotional intimacy
        - Handle conflicts and stress
        - View yourself and others
        
        Understanding your attachment style can help you develop more secure relationships.
        """)
    
    # Actions section
    st.subheader("Next Steps")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Take the Test Again", type="primary"):
            # Reset session state to restart the test
            st.session_state.page = 0
            st.session_state.answers = {}
            st.session_state.test_done = False
            st.rerun()
    
    with col2:
        if st.download_button(
            label="📊 Download Results",
            data=pd.DataFrame({
                'Attachment Style': st.session_state.attachment_styles,
                'Percentage': st.session_state.pred_percentages
            }).to_csv().encode('utf-8'),
            file_name='attachment_style_results.csv',
            mime='text/csv',
        ):
            st.success("Results downloaded!")

# Footer
st.markdown("""---
*This app is for educational purposes only and not a substitute for professional advice.*
""")