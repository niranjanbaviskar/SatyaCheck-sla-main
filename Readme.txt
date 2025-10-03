# 🔍 SatyaCheck

## News Credibility Verification through Social Media Cross-Referencing

SatyaCheck is a powerful Streamlit application designed to verify news credibility by cross-referencing content with real-time social media data. The name "Satya" means "truth" in Sanskrit, reflecting our commitment to reliable information.

![SatyaCheck Banner](https://via.placeholder.com/800x200.png?text=SatyaCheck)

## 📋 Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API Requirements](#api-requirements)
- [User Interface](#user-interface)
- [Profession-Specific Analysis](#profession-specific-analysis)
- [Data Processing](#data-processing)
- [Performance Optimizations](#performance-optimizations)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multi-Platform Analysis**: Searches across Reddit and YouTube for related content
- **Advanced NLP**: Extracts meaningful keywords from news articles
- **Profession-Specific Insights**: Customized analysis based on user's professional background
- **Visual Analytics**: Interactive charts and visualizations of results
- **Detailed Post Analysis**: Shows keyword matches, post statistics, and credibility metrics
- **Real-time Processing**: Concurrent API calls with caching for efficient performance
- **Data Export**: Export complete results in CSV format
- **User Feedback System**: Integrated feedback collection for continuous improvement
- **Customizable Parameters**: Adjust search depth, keyword sensitivity, and platforms

## 🔄 How It Works

1. **Text Input**: User pastes the news article text
2. **Keyword Extraction**: System identifies important keywords using NLP
3. **Social Media Search**: Concurrent searches across selected platforms (Reddit, YouTube)
4. **Credibility Assessment**: Analyzes matches against social media content
5. **Sensationalism Analysis**: Evaluates language indicators of potential fake news
6. **Profession-Specific Evaluation**: Applies specialized heuristics based on user's profession
7. **Result Visualization**: Presents findings through interactive charts and data tables
8. **Report Generation**: Provides downloadable reports with complete analysis

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/satyacheck.git
cd satyacheck

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with the following packages:

```
streamlit
firebase-admin
requests
beautifulsoup4
spacy
praw
google-api-python-client
scikit-learn
pandas
numpy
plotly
cachetools
```

You'll also need to download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## 🔑 API Requirements

SatyaCheck requires API credentials for both Reddit and YouTube. You'll need to:

1. **Reddit API**:
   - Create an app at https://www.reddit.com/prefs/apps
   - Set up a script type application
   - Use the client ID, client secret, and user agent in the app

2. **YouTube API**:
   - Get an API key from the Google Cloud Console
   - Enable the YouTube Data API v3

3. **Firebase** (for logging):
   - Create a Firebase project
   - Generate a service account key
   - Save it as `firebase_credentials.json` in your project directory

## 🖥️ Usage

Run the application with:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to access the interface.

### Basic Workflow

1. Paste news article text in the provided area
2. Adjust analysis settings in the sidebar if needed
3. Click "Analyze Text"
4. Review the credibility assessment and matched social media posts
5. Download CSV reports for further analysis
6. Provide feedback to help improve the system

## 🎨 User Interface

SatyaCheck features a modern, user-friendly interface with:

- **Clean Card-Based Layout**: Easy to navigate results
- **Interactive Charts**: Visualize platform distribution and keyword matches
- **Color-Coded Indicators**: Quickly interpret credibility scores
- **Responsive Design**: Works on desktop and mobile devices
- **Customizable Settings**: Adjust analysis parameters from the sidebar

## 👥 Profession-Specific Analysis

SatyaCheck provides tailored analysis for various professional backgrounds:

- Traders/Investors
- Journalists
- Political Analysts
- Government Officials & Policy Makers
- Researchers/Academics
- Media and Broadcasting Organizations
- Fact-Checking Organizations
- Public Relations & Communications Professionals
- Marketing & Advertising Professionals
- Legal Professionals
- Corporate Communications Teams
- Social Media Analysts
- Risk Management Specialists

Each profession has specialized thresholds and commentary to match specific needs.

## 🔄 Data Processing

### Keyword Extraction

SatyaCheck uses spaCy to identify:
- Named entities (people, organizations, locations, events)
- Noun phrases
- Important verbs and nouns

### Post Matching

The system calculates match percentages based on:
- Keyword presence in social media posts
- Weighted importance of keywords
- Temporal relevance of posts

### Credibility Assessment

Credibility is determined through:
- Maximum keyword match percentage
- Number of supporting posts
- Sensationalism indicators
- Platform-specific metrics

## ⚡ Performance Optimizations

- **Concurrent API Calls**: ThreadPoolExecutor for parallel data fetching
- **TTL Cache**: Caching results for 5 minutes to reduce API calls
- **Efficient Data Processing**: Streamlined text analysis pipeline
- **Firebase Integration**: Non-blocking activity logging

## ⚠️ Limitations

- API rate limits may affect search depth
- Analysis is limited to public posts on selected platforms
- Transformer model for fake news detection is currently a stub implementation
- Results depend on social media coverage of the topic

## 🚀 Future Enhancements

- Expand platform coverage (Twitter, Facebook, etc.)
- Implement full transformer-based fake news detection
- Add sentiment analysis of matching posts
- Create user accounts to save and compare analyses
- Develop browser extension for one-click analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: SatyaCheck is part of an evolving media literacy toolkit. Always verify information with multiple reliable sources.