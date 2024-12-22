# 🏥 Medical Guidelines Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system that helps healthcare practitioners quickly access, interpret, and apply complex medical guidelines in their clinical practice. Built with LangChain, Streamlit, and powered by Llama 3.3 70B.

![Medical Guidelines Assistant](https://raw.githubusercontent.com/username/medical-guidelines-assistant/main/docs/assistant-demo.gif)

## 🎯 Problem Statement

Healthcare practitioners face several critical challenges when working with medical guidelines:

### Current Challenges
1. **Time Constraints**
   - Average consultation time: 10-15 minutes
   - Traditional guideline lookup: 5-10 minutes
   - Multiple conditions require checking multiple guidelines
   - Urgent decisions need quick access to accurate information

2. **Guideline Complexity**
   - Guidelines can be 100+ pages long
   - Multiple updates throughout the year
   - Complex decision trees and conditions
   - Interconnected recommendations across different conditions

3. **Real-world Scenarios**
   - Patients often have multiple conditions requiring cross-referencing
   - Guidelines can have conflicting recommendations
   - Different evidence levels for different recommendations
   - Need to consider patient-specific contraindications

### Impact of These Challenges
- Delayed decision-making
- Increased risk of overlooking important contraindications
- Difficulty staying updated with latest changes
- Reduced time for patient interaction
- Potential for missing critical information

## 💡 Solution

This basic Medical Guidelines Assistant addresses could be the base of an application that can truly tackle these challenges.

### Real-world Use Cases
1. **Complex Patient Scenarios**
   ```
   Query: "Treatment options for heart failure patient with diabetes and reduced kidney function"
   Response: Comprehensive guidance considering all three conditions with specific guideline citations
   ```

2. **Emergency Situations**
   ```
   Query: "Acute heart failure management with concurrent COVID"
   Response: Rapid, prioritized recommendations with clear evidence levels
   ```

   # 🏥 Medical Guidelines Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system that helps healthcare practitioners quickly access, interpret, and apply complex medical guidelines in their clinical practice. Built with LangChain, Streamlit, and powered by Llama 3.3 70B.

![Medical Guidelines Assistant](https://raw.githubusercontent.com/username/medical-guidelines-assistant/main/docs/assistant-demo.gif)

## 🌟 Key Features

- **Guideline Search**: Instantly access medical guidelines using natural language queries
- **Context-Aware Responses**: Considers chat history and multiple guidelines simultaneously
- **Source Attribution**: Every recommendation includes specific guideline sources and evidence levels

## 📚 Supported Guidelines

- AHA/ACC/HFSA 2022 Heart Failure Management Guidelines
- NICE Guidelines
- WHO Clinical Protocols
- CDC Clinical Practice Guidelines
- Specialty-specific society guidelines

## 🛠️ Technical Architecture

### Stack
- **LLM**: Llama 3.3 70B (via Together AI)
- **Embeddings**: Mistral AI
- **Vector Store**: Pinecone (Serverless)
- **Framework**: LangChain
- **Frontend**: Streamlit

### System Components
1. **Vector Store Setup**
   - PDF document ingestion
   - Recursive text splitting
   - Document embedding using Mistral AI
   - Serverless Pinecone index management

2. **RAG Pipeline**
   - History-aware retrieval chain
   - Contextualized question processing
   - Stream-enabled response generation
   - Chat history management

## 🚀 Getting Started

### Prerequisites
```bash
python 3.8+
pip
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/username/medical-guidelines-assistant.git
cd medical-guidelines-assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
TOGETHER_API_KEY=your_together_api_key
MISTRALAI_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

4. Run the application
```bash
streamlit run app.py
```

## 🏗️ Project Structure

```
medical-guidelines-assistant/
├── app.py                 # Streamlit interface
├── src/
│   ├── chain_builder.py   # RAG chain configuration
│   ├── config.py         # System configuration
│   └── utils.py          # Vector store utilities
├── Guidelines_Data/       # PDF guidelines storage
└── requirements.txt      # Dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/username/medical-guidelines-assistant](https://github.com/username/medical-guidelines-assistant)