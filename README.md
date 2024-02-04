# Programmed Differently
Associated with the JCMC paper "Programmed Differently? Testing for Gender Differences in Python Programming Style and Quality on GitHub" DOI: https://doi.org/10.1093/jcmc/zmad049

🌟 Research Highlights 🌟
(I) I deploy a methodology to conduct a computational analysis of the code itself. Computer code in open-source is a text that is read, understood and shared. 
(II) I approach code as a form of communication; where cultural norms of gender affect the interpretation of its quality and value. This means that the same code may be interpreted differently if it is assumed that the author is a man or a woman.
(III) I extend work on automated gender inference with an analysis that includes the categorisation of gender beyond a binary. I differentiate between "anonymous" and "ambiguous" users, distinguishing between users with mixed gender markers and those with none. 

🌟 Key Discoveries 🌟
(I) Style differences in Python coding exist across genders, but not in overall quality.
(II) Using Random Forests, the study predicts a coder's inferred gender based on their coding style.
(III) Similarities between women's and anonymous users' coding styles suggest strategies to avoid sexism on technical platforms.
(IV) Implications for AI development and generative coding tools like ChatGPT due to gendered coding styles.

🛠 Extra Resources 🛠
(I) Interactive website for testing your code's gendered style: https://www.sianbrooke.co.uk/programmed-differently

### Abstract
The underrepresentation of women in open-source software is frequently attributed to women’s lack of innate aptitude compared to men: a natural “gender difference” in technical ability. Approaching code as a form of communication, I conduct a novel empirical study of gender difference in Python programming on GitHub. Based on 1,728 open-source projects, I ask if there is a gender difference in the quality and style of Python code measured in adherence to PEP-8 guidelines. I found significant gender differences in structure and how Python files are organised. Whilst there is gendered variation in programming style, there is no evidence of gender difference in code quality. Using a Random Forest model, I show that the gender of a programmer can be predicted from the style of their Python code. The study concludes that gender differences in Python code are a matter of style, not quality.

### Lay Summary 
This study examines whether there is a difference in Python programming styles between gender groups. I examine available code on GitHub, a cloud-based hosting platform for collaboration known as version control, often used in open-source software development. First, I infer the gender of users from their usernames and the information provided on their profiles, labelling users as feminine, masculine, ambiguous, and anonymous. Anonymous users had no gender-based markers on their profiles, while ambiguous users had feminine and masculine characteristics. I then collect the publicly available projects of these users written in Python. Next, I analyse and generate statistics on Python files’ adherence to style guidelines using a linter, an automated checking of source code for programmatic and stylistic errors. My findings reveal a gendered difference in the structure and components of Python files. However, I also discovered no gender difference regarding violations of Python style guidelines and code quality. This study shows gender difference in Python programming styles but not in the standard or quality of the code. 

Interactive site: https://www.sianbrooke.com/programmed-differently.


