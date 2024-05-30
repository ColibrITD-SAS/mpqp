from mpqp.execution.connection.env_manager import get_env_variable


def get_google_account_info() -> str:
    """
    Get the IonQ API key from the environment variables.

    Returns:
        str: A string containing the IonQ API key.
    """
    ionq_api_key = get_env_variable("IONQ_API_KEY")

    return f"""   IONQ_API_KEY : '{ionq_api_key}'"""



def get_all_job_ids() -> list[str]:
    """
    Retrieves all job IDs associated with google jobs.

    Returns:
        A list of job IDs.
    
    #3M-TODO: get job of google
    """
    return []


