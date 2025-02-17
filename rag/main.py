import langchain as lc

class AutoRepoAnalyzer:
    """Class Construct for Automatically Analyzing Repositories for creating a md file"""
    def __init__(self, repo_path, repo_name, repo_type, repo_url, repo, filetypes):
        """Initialization method for building up the auto analyzer tool."""
        self.repo_path = repo_path
        self.repo_name = repo_name
        self.repo_type = repo_type
        self.repo_url = repo_url    
        self.repo = repo
        self.filetypes = filetypes
        self.input = ""
        self.output = ""
        self.status = False
    
    def set_input(self, input):
        """Method to set the input for the auto analyzer tool."""
        self.input = input

    def get_output(self):
        """Method to get the output from the auto analyzer tool."""
        return self.output
    
    def get_reponse_status(self):
        """Method to get the response from the system for creating a summarize file."""
        return self.status
