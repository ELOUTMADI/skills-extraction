import openai
import json
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already done
nltk.download('punkt')

def extract_skills_from_job_description(job_description, api_key=None):
    """
    Extracts a list of unique technical skills from a job description.

    Parameters:
    - job_description (str): The job description text.
    - api_key (str, optional): Your OpenAI API key. If not provided, it will look for 'OPENAI_API_KEY' in environment variables.

    Returns:
    - list: A list of unique technical skills extracted from the job description.
    """

    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Tokenize the job description into sentences
    sentences = sent_tokenize(job_description)
    skills_list = []

    for sentence in sentences:
        # Check if the sentence contains technical skills
        if check_sentence_for_skills(sentence):
            # Extract skills from the sentence
            extracted_skills = extract_skills_from_sentence(sentence)
            skills_list.extend(extracted_skills)

    # Remove duplicates and return
    unique_skills = list(set(skills_list))
    return unique_skills

def check_sentence_for_skills(sentence):
    """
    Determines whether a sentence contains any mention of technical skills.

    Parameters:
    - sentence (str): The sentence to check.

    Returns:
    - bool: True if skills are mentioned, False otherwise.
    """
    skill_check_function = {
        "name": "check_for_skills",
        "description": "Determines whether a sentence contains any mention of skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "contains_skills": {
                    "type": "boolean",
                    "description": "True if skills are mentioned, False otherwise."
                }
            },
            "required": ["contains_skills"]
        }
    }

    response_check = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that determines whether a sentence contains any mention of technical skills required for a job."
            },
            {
                "role": "user",
                "content": f"Does the following sentence mention any job-related technical skills? Answer with true or false.\n\nSentence: \"{sentence}\""
            }
        ],
        functions=[skill_check_function],
        function_call={"name": "check_for_skills"}
    )

    # Parse the response
    check_result = response_check['choices'][0]['message']['function_call']['arguments']
    contains_skills = json.loads(check_result)['contains_skills']
    return contains_skills

def extract_skills_from_sentence(sentence):
    """
    Extracts technical skills mentioned in a sentence.

    Parameters:
    - sentence (str): The sentence to extract skills from.

    Returns:
    - list: A list of skills extracted from the sentence.
    """
    extract_skills_function = {
        "name": "extract_skills",
        "description": "Extracts skills mentioned in a sentence.",
        "parameters": {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of skills extracted from the sentence."
                }
            },
            "required": ["skills"]
        }
    }

    response_extract = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that extracts only technical skills, tools, technologies, and methodologies mentioned in a sentence. Ignore soft skills or non-technical requirements."
            },
            {
                "role": "user",
                "content": f"Extract the technical skills from the following sentence:\n\nSentence: \"{sentence}\""
            }
        ],
        functions=[extract_skills_function],
        function_call={"name": "extract_skills"}
    )

    # Parse the response
    extract_result = response_extract['choices'][0]['message']['function_call']['arguments']
    extracted_skills = json.loads(extract_result)['skills']
    return extracted_skills

if __name__ == "__main__":
    # Example job description
    
# Open and read the JSON file
    with open('job_data.json', 'r') as file:
        job_description = json.load(file)[27]['Job Description']
    job_description = """
    Le pôle IT du Groupe poursuit son développement avec la création d’un nouveau business line en Système d’information Télécom.

    MISSIONS :

    Développer des solutions en mode agile pour répondre aux attentes business.
    Développement du Framework TEDh.
    Développement en PySpark, shell script, HQL, de nouvelles fonctionnalités du framework d'ingestion.
    Gestion de la montée de version Spark vers la 2.2.
    Paramétrage des transformations dans le lakeshore.
    Revue de code.
    Assurer le support lors des grandes opérations de migration des données.
    Poste de gestion des évolutions et MCO(80/20).

    Profil recherché : 

    Formation : Bac + 5 (Master ou école d’ingénieur) en système d’information ;

    Expérience : Minimum 3 ans dans un poste ;

    Vous avez pratiqué des projets de développement Data ;
    Excellent relationnel, qualités de communication et d’écoute en français ;
    Bonne organisation et gestion des priorités ;
    Expérience significative en environnement back-end Python et/ou Scala 
    Bonne maitrise de SNOWFLAKE
    Bonne connaissance en SQL, POO, Spark, Hadoop.
    Capable de comprendre et d’écrire du code python structuré, testé.
    Utilise les bonnes pratiques de développement (tests automatisés, revue de code, clean code, intégration continue)
    Grande capacité d'adaptation, aime travailler en équipe (pair programming) et partage les bonnes pratiques DevOps/DataOps
    Connaissance d'un outil d'ordonnancement (type Airflow)

    ENVIRONNEMENT TECHNIQUE 

    Vous serez amenés à travailler sous les environnements suivants :

    Système d’exploitation : Hadoop Cloudera Cloudera CDP 5.16
    Langage de programmation : PySpark, shell script, HQL, Python, SQL.
    Outils : Gitlab, Framework TEDH (dec interne PySpark), PyCharm 
    Système de gestion de BDD: Haddoop Distribution Cloudera CDP (HDFS) 
    Infrastructure / Socle logicielle : Plateforme Big Data HP 
    Snowflake : Compétence obligatoire
    Autres : Agile, BigData, Hadoop…
    """

    # Replace 'your-api-key' with your actual OpenAI API key or set the 'OPENAI_API_KEY' environment variable
    openai.api_key = ''

    skills = extract_skills_from_job_description(job_description, openai.api_key)
    print("Extracted Skills:", skills)