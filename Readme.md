🚀 AI Integrated API Documentation Generator:
An automated tool that generates and maintains documentation for your API endpoints by analyzing git commit changes. The system automatically detects modifications to controller files and creates or updates corresponding documentation.


🌟 Features:
Automated Documentation: Generates API documentation from code changes in real-time
Git Integration: Hooks into your git workflow via webhooks
Smart Diffing: Intelligently detects added, modified, and deleted API endpoints
Markdown Output: Creates clean, well-structured documentation in Markdown format
Method-Level Tracking: Uses unique markers to track and update individual endpoint documentation


📋 Prerequisites:
Python 3.7+
Git repository containing API controllers
Hugging Face API key for the Mistral-7B-Instruct model


🔧 Installation
Clone this repository:
    git clone https://github.com/yourusername/api-doc-generator.git
    cd api-doc-generator

Install required dependencies:
    pip install -r requirements.txt

Create a .env file with the following variables:
    REPO_PATH=/path/to/your/git/repository
    DOCS_PATH=/path/to/output/documentation
    WEBHOOK_SECRET=your_github_webhook_secret
    HF_API_KEY=your_huggingface_api_key


🚀 Usage
Running the server
    python app.py
    This starts a Flask server listening for webhook events. By default, it runs on http://127.0.0.1:8000.

Setting up the webhook
    Go to your GitHub repository settings
    Navigate to Webhooks > Add webhook
    Set the Payload URL to your server's URL (e.g., https://your-server.com/webhook)
    Set Content type to application/json
    Enter your secret in the Secret field (same as WEBHOOK_SECRET in .env)
    Select "Just the push event" for triggering the webhook
    Enable the webhook

Testing locally
    To test the system without setting up a webhook, modify and uncomment the following line in app.py:
        # test_locally("YOUR_COMMIT_HASH")  # Replace with your commit hash


💡 How It Works
Change Detection: The system receives a webhook notification when code is pushed to the repository
API Analysis: It identifies controller files and extracts API methods using regex patterns
Documentation Generation: For each changed method, it generates documentation using the Mistral-7B LLM
Documentation Update: It creates or updates Markdown files with the generated documentation


📝 Documentation Format
The system generates documentation with the following structure:
    #ControllerName API Documentation

    <!-- START_METHOD: MethodName -->
    ## MethodName

    ### Description
    Detailed description based on the code logic.

    ### Parameters
    - param1 (type): Description
    - param2 (type): Description

    ### Logic
    Line-by-line explanation of the method implementation.
    <!-- END_METHOD: MethodName -->


⚙️ Configuration
Adjust the following parameters in the .env file:

    REPO_PATH: Path to your git repository
    DOCS_PATH: Path where documentation files will be stored
    WEBHOOK_SECRET: Secret for validating GitHub webhook requests
    HF_API_KEY: Hugging Face API key for accessing the Mistral model


🔒 Security Notes

Keep your Hugging Face API key and webhook secret secure
Consider restricting which repositories and branches can trigger documentation updates


🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.