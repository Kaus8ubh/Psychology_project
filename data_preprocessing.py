import pandas as pd
import numpy as np  

df = pd.read_csv('synthetic_data.csv')
# print(df.columns.tolist())
authoritative_ps = [
    "My parents are responsive to my feelings and needs:   ",
    "My parents consider my wishes before asking me to do something: ",
    "My parents explain how they feel about my good/bad behaviour.",
    "My parents encourage me to talk about my feelings and problems.",
    "My parents encourage me to freely express my thoughts, even if I disagree with them.",
    "My parents explain the reasons behind their expectations.",
    "My parents provide comfort and understanding when I am upset.",
    "My parents compliment me.",
    "My parents consider my preferences when making family plans (e.g., weekends away and holidays).",
    "My parents respect my opinion and encourage me to express it.",
    "My parents treat me as an equal member of the family.",
    "My parents provide reasons for their expectations of me.",
    "My parents and I have warm and intimate times together."
]

authoritarian_ps = [
    "When I ask my parents why I have to do something, they tell me it's because they said so or because they want it that way.",
    "My parents punish me by taking away privileges (e.g., TV, games, visiting friends).",
    "My parents yell when they disapprove of my behavior.",
    "My parents explode in anger towards me.",
    "My parents spank me when they don't like what I do or say.",
    "My parents use criticism to make me improve my behavior.",
    "My parents use threats as a form of punishment with little or no justification.",
    "My parents punish me by withholding emotional expressions (e.g., kisses and cuddles).   ",
    "My parents openly criticize me when my behavior does not meet their expectations.   ",
    "My parents struggle to change how I think or feel about things.  ",
    "My parents point out my past behavioral problems to ensure I don't repeat them.   ",
    "My parents remind me that they are my parents.   ",
    "My parents remind me of all the things they do for me.   "
]

permissive_ps = [
    "My parents find it difficult to discipline me.   ",
    "My parents give in to me when I cause a commotion about something.   ",
    "My parents spoil me.   ",
    "My parents ignore my bad behavior.   "
]

secure_as = [
    "I don't usually feel hesitant and demeaned discussing my problems with others.",
    "It is not difficult for me to get emotionally close to others.",
    "I find it very easy to depend on others.   ",
    "I prefer to express my feelings.   ",
    "I find people trustworthy.   ",
    "I usually discuss my problems with my relatives and friends.   ",
    "I share almost everything with my closed ones.",
    "I find it relatively easy to get close to others.   ",

]

avoidant_as = [
    "I do not find it easy in being close to others.",
    " I am least interested in being attached with others.",
    "I don't prefer people getting too close to me.   ",
    "I don't worry if others don't accept me.",
    "I withdraw myself to get too close to others.   ",
    "Unlike others, I am usually unwilling to get closer.",
    "The moment someone starts to get closer, I find myself pulling away.   ",
    "I don't bother, if closed ones like or love me.  ",
    "I am comfortable, irrespective of people being with me or not.  ",
    "I feel shy in sharing things to others.   "
]

anxious_as = [
    "I worry about being alone.",
    "At times, I feel like getting close to others, but then something draws me back.  ",
    "I feel worry about being abandoned by others.  ",
    "I get annoyed when people are unavailable at the time of need.",
    "I feel depressed if someone close to me is unavailable at the time of need.",
    "It is worrying for me, if others neglect me.   ",
    "I get depressed when my closed ones are not around me as much as I would like.  ",
    "I sometimes worry that I will be hurt if I allow myself to become too close to others. ",
    "I become irritated or upset in maintaining the relationship with others.  "
]

# parenting style 
df_authoritative = df[authoritative_ps]
df_authoritarian = df[authoritarian_ps]
df_permissive = df[permissive_ps]
# attachment style
df_secure = df[secure_as]
df_avoidant = df[avoidant_as]
df_anxious = df[anxious_as]

reversed_questions = [
    "I don't worry if others don't accept me.",
    "The moment someone starts to get closer, I find myself pulling away.   ",
    "I don't bother, if closed ones like or love me.  ",
    "I am comfortable, irrespective of people being with me or not.  ",
    "I feel shy in sharing things to others.   "
]

for col in reversed_questions:
    df_avoidant.loc[:, col] = 6 - df_avoidant[col]

# Calculate mean score across all authoritative parenting questions
df['authoritative_score'] = df_authoritative.mean(axis=1)
df['authoritarian_score'] = df_authoritarian.mean(axis=1)
df['permissive_score'] = df_permissive.mean(axis=1)
df['secure_score'] = df_secure.mean(axis=1)
df['avoidant_score'] = df_avoidant.mean(axis=1)
df['anxious_score'] = df_anxious.mean(axis=1)

score_df = df[['authoritative_score', 'authoritarian_score', 'permissive_score', 'secure_score', 'avoidant_score', 'anxious_score']]

# export the scores to a new CSV file
score_df.to_csv('scores.csv', index=False)
print("Scores have been calculated and saved to scores.csv")