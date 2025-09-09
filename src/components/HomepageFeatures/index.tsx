import type {ReactNode} from 'react';
import React, { useState } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type LibraryItem = {
  name: string;
  icon: string;
  description: string;
  website: string;
  color: string;
};

type FeatureItem = {
  title: string;
  description: ReactNode;
  link: string;
  libraries: LibraryItem[];
  gradient: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Machine Learning Lifecycle',
    description: (
      <>
        Complete ML lifecycle from data collection to deployment. Traditional machine learning 
        approaches with comprehensive guides for supervised, unsupervised, and reinforcement learning.
      </>
    ),
    link: '/machine-learning-lifecycle',
    gradient: 'linear-gradient(135deg, #48BB78 0%, #38A169 100%)',
    libraries: [
      { name: 'Scikit-learn', icon: '🔬', description: 'Machine Learning Library', website: 'https://scikit-learn.org', color: '#F7931E' },
      { name: 'TensorFlow', icon: '🧠', description: 'Deep Learning Framework', website: 'https://tensorflow.org', color: '#FF6F00' },
      { name: 'PyTorch', icon: '🔥', description: 'Deep Learning Framework', website: 'https://pytorch.org', color: '#EE4C2C' },
      { name: 'Pandas', icon: '🐼', description: 'Data Analysis Library', website: 'https://pandas.pydata.org', color: '#150458' },
      { name: 'NumPy', icon: '🔢', description: 'Numerical Computing', website: 'https://numpy.org', color: '#013243' },
      { name: 'Matplotlib', icon: '📊', description: 'Data Visualization', website: 'https://matplotlib.org', color: '#11557C' },
      { name: 'Seaborn', icon: '📈', description: 'Statistical Visualization', website: 'https://seaborn.pydata.org', color: '#4C72B0' },
      { name: 'XGBoost', icon: '🚀', description: 'Gradient Boosting', website: 'https://xgboost.ai', color: '#FF6B35' },
      { name: 'LightGBM', icon: '💡', description: 'Gradient Boosting', website: 'https://lightgbm.readthedocs.io', color: '#4285F4' },
      { name: 'CatBoost', icon: '🐱', description: 'Gradient Boosting', website: 'https://catboost.ai', color: '#FFCA28' },
      { name: 'Keras', icon: '🧩', description: 'Deep Learning API', website: 'https://keras.io', color: '#D00000' },
      { name: 'Jupyter', icon: '📓', description: 'Interactive Computing', website: 'https://jupyter.org', color: '#F37626' },
      { name: 'MLflow', icon: '🔄', description: 'ML Lifecycle Management', website: 'https://mlflow.org', color: '#0194E2' },
      { name: 'Optuna', icon: '⚡', description: 'Hyperparameter Optimization', website: 'https://optuna.org', color: '#5DADE2' },
      { name: 'Weights & Biases', icon: '📊', description: 'Experiment Tracking', website: 'https://wandb.ai', color: '#FFBE00' },
      { name: 'Apache Spark', icon: '⚡', description: 'Big Data Processing', website: 'https://spark.apache.org', color: '#E25A1C' },
      { name: 'Dask', icon: '⚖️', description: 'Parallel Computing', website: 'https://dask.org', color: '#FC4C02' },
      { name: 'Plotly', icon: '📊', description: 'Interactive Visualization', website: 'https://plotly.com', color: '#3F4F75' },
      { name: 'Streamlit', icon: '🎯', description: 'Data Apps Framework', website: 'https://streamlit.io', color: '#FF4B4B' },
      { name: 'FastAPI', icon: '🚀', description: 'API Framework', website: 'https://fastapi.tiangolo.com', color: '#009688' }
    ]
  },
  {
    title: 'Natural Language Processing',
    description: (
      <>
        End-to-end NLP pipeline covering text preprocessing, feature engineering, model training,
        and deployment for sentiment analysis, classification, and language understanding tasks.
      </>
    ),
    link: '/nlp-lifecycle',
    gradient: 'linear-gradient(135deg, #4299E1 0%, #3182CE 100%)',
    libraries: [
      { name: 'Hugging Face', icon: '🤗', description: 'Transformers Library', website: 'https://huggingface.co', color: '#FF6B6B' },
      { name: 'spaCy', icon: '🌶️', description: 'Industrial NLP', website: 'https://spacy.io', color: '#09A3D5' },
      { name: 'NLTK', icon: '🦅', description: 'Natural Language Toolkit', website: 'https://nltk.org', color: '#2E8B57' },
      { name: 'OpenAI', icon: '🤖', description: 'GPT Models API', website: 'https://openai.com', color: '#10A37F' },
      { name: 'Gensim', icon: '📚', description: 'Topic Modeling', website: 'https://radimrehurek.com/gensim/', color: '#4B8BBE' },
      { name: 'TextBlob', icon: '💬', description: 'Simple Text Processing', website: 'https://textblob.readthedocs.io', color: '#646569' },
      { name: 'Transformers', icon: '🔄', description: 'State-of-the-art NLP', website: 'https://huggingface.co/transformers', color: '#FF9500' },
      { name: 'BERT', icon: '🧠', description: 'Bidirectional Encoder', website: 'https://github.com/google-research/bert', color: '#4285F4' },
      { name: 'Stanford NLP', icon: '🎓', description: 'CoreNLP Toolkit', website: 'https://stanfordnlp.github.io/CoreNLP/', color: '#8C1515' },
      { name: 'Flair', icon: '✨', description: 'NLP Framework', website: 'https://github.com/flairNLP/flair', color: '#FF6B35' },
      { name: 'AllenNLP', icon: '🔬', description: 'Research Library', website: 'https://allennlp.org', color: '#1BA1E2' },
      { name: 'Stanza', icon: '📝', description: 'Python NLP Library', website: 'https://stanfordnlp.github.io/stanza/', color: '#8C1515' },
      { name: 'Polyglot', icon: '🌍', description: 'Multilingual NLP', website: 'https://polyglot.readthedocs.io', color: '#FFA500' },
      { name: 'Rasa', icon: '🤖', description: 'Conversational AI', website: 'https://rasa.com', color: '#5A67D8' },
      { name: 'ChatterBot', icon: '💬', description: 'Machine Learning Chatbot', website: 'https://chatterbot.readthedocs.io', color: '#FF6B6B' },
      { name: 'Langchain', icon: '⛓️', description: 'LLM Applications Framework', website: 'https://langchain.com', color: '#FF4B4B' },
      { name: 'LlamaIndex', icon: '🦙', description: 'Data Framework for LLMs', website: 'https://llamaindex.ai', color: '#B794F6' },
      { name: 'Sentence Transformers', icon: '🔗', description: 'Sentence Embeddings', website: 'https://www.sbert.net', color: '#00D2FF' },
      { name: 'Whisper', icon: '🎤', description: 'Speech Recognition', website: 'https://openai.com/blog/whisper/', color: '#10A37F' },
      { name: 'Cohere', icon: '🧠', description: 'Language AI Platform', website: 'https://cohere.com', color: '#FF6B35' }
    ]
  },
  {
    title: 'Computer Vision',
    description: (
      <>
        Complete computer vision workflow from image data collection to model deployment.
        Object detection, classification, segmentation, and real-time vision applications.
      </>
    ),
    link: '/computer-vision-lifecycle',
    gradient: 'linear-gradient(135deg, #ED8936 0%, #DD6B20 100%)',
    libraries: [
      { name: 'OpenCV', icon: '👁️', description: 'Computer Vision Library', website: 'https://opencv.org', color: '#5C3EE8' },
      { name: 'YOLO', icon: '⚡', description: 'Real-time Object Detection', website: 'https://pjreddie.com/darknet/yolo/', color: '#00FFFF' },
      { name: 'Detectron2', icon: '🔍', description: 'Object Detection Platform', website: 'https://detectron2.readthedocs.io', color: '#1C4E80' },
      { name: 'Pillow (PIL)', icon: '🖼️', description: 'Image Processing', website: 'https://pillow.readthedocs.io', color: '#306998' },
      { name: 'ImageIO', icon: '📸', description: 'Image I/O Library', website: 'https://imageio.readthedocs.io', color: '#FF6B35' },
      { name: 'Scikit-Image', icon: '🔬', description: 'Image Processing', website: 'https://scikit-image.org', color: '#F7931E' },
      { name: 'Albumentations', icon: '🎨', description: 'Image Augmentation', website: 'https://albumentations.ai', color: '#FF4081' },
      { name: 'Torchvision', icon: '🔥', description: 'Computer Vision for PyTorch', website: 'https://pytorch.org/vision/', color: '#EE4C2C' },
      { name: 'TensorFlow CV', icon: '🧠', description: 'Computer Vision with TF', website: 'https://www.tensorflow.org/tutorials/images', color: '#FF6F00' },
      { name: 'Mediapipe', icon: '🤲', description: 'Perception Pipeline', website: 'https://mediapipe.dev', color: '#4285F4' },
      { name: 'MMDetection', icon: '🎯', description: 'Detection Toolbox', website: 'https://mmdetection.readthedocs.io', color: '#FF6B35' },
      { name: 'Ultralytics', icon: '🚀', description: 'YOLOv8 Implementation', website: 'https://ultralytics.com', color: '#00D2FF' },
      { name: 'Roboflow', icon: '🤖', description: 'Computer Vision Pipeline', website: 'https://roboflow.com', color: '#6366F1' },
      { name: 'Supervision', icon: '👀', description: 'Computer Vision Utils', website: 'https://supervision.roboflow.com', color: '#8B5CF6' },
      { name: 'ImageAI', icon: '🧠', description: 'Easy Computer Vision', website: 'https://imageai.readthedocs.io', color: '#10B981' },
      { name: 'FiftyOne', icon: '📊', description: 'Dataset Management', website: 'https://fiftyone.ai', color: '#FF6B6B' },
      { name: 'Label Studio', icon: '🏷️', description: 'Data Labeling', website: 'https://labelstud.io', color: '#FF4B4B' },
      { name: 'CVAT', icon: '🎯', description: 'Annotation Tool', website: 'https://cvat.org', color: '#1890FF' },
      { name: 'DeepFace', icon: '😊', description: 'Face Recognition', website: 'https://github.com/serengil/deepface', color: '#FF6B35' },
      { name: 'Gradio', icon: '🎛️', description: 'ML Interface Builder', website: 'https://gradio.app', color: '#FF7C00' }
    ]
  },
  {
    title: 'Generative AI',
    description: (
      <>
        Comprehensive guide for generative AI systems including LLMs, diffusion models,
        and multimodal generation with safety, evaluation, and deployment considerations.
      </>
    ),
    link: '/generative-ai-lifecycle',
    gradient: 'linear-gradient(135deg, #9F7AEA 0%, #805AD5 100%)',
    libraries: [
      { name: 'OpenAI GPT', icon: '🤖', description: 'Large Language Models', website: 'https://openai.com', color: '#10A37F' },
      { name: 'Anthropic Claude', icon: '🎭', description: 'Constitutional AI', website: 'https://anthropic.com', color: '#D4A574' },
      { name: 'Stable Diffusion', icon: '🎨', description: 'Text-to-Image Generation', website: 'https://stability.ai', color: '#FF6B6B' },
      { name: 'DALL-E', icon: '🖼️', description: 'AI Image Generation', website: 'https://openai.com/dall-e-2/', color: '#10A37F' },
      { name: 'Midjourney', icon: '🌟', description: 'AI Art Generation', website: 'https://midjourney.com', color: '#5865F2' },
      { name: 'Hugging Face Hub', icon: '🤗', description: 'Model Repository', website: 'https://huggingface.co', color: '#FF6B6B' },
      { name: 'LangChain', icon: '⛓️', description: 'LLM Application Framework', website: 'https://langchain.com', color: '#FF4B4B' },
      { name: 'LlamaIndex', icon: '🦙', description: 'Data Framework for LLMs', website: 'https://llamaindex.ai', color: '#B794F6' },
      { name: 'AutoGPT', icon: '🔄', description: 'Autonomous AI Agent', website: 'https://github.com/Significant-Gravitas/AutoGPT', color: '#FF6B35' },
      { name: 'Google Bard', icon: '🎵', description: 'Conversational AI', website: 'https://bard.google.com', color: '#4285F4' },
      { name: 'Cohere', icon: '🧠', description: 'Language AI Platform', website: 'https://cohere.com', color: '#FF6B35' },
      { name: 'Runway ML', icon: '🎬', description: 'Creative AI Tools', website: 'https://runwayml.com', color: '#00FF88' },
      { name: 'DeepL', icon: '🌐', description: 'AI Translation', website: 'https://deepl.com', color: '#0F2027' },
      { name: 'Replicate', icon: '🔄', description: 'Run ML Models in Cloud', website: 'https://replicate.com', color: '#FF4B4B' },
      { name: 'Pinecone', icon: '🌲', description: 'Vector Database', website: 'https://pinecone.io', color: '#8B5CF6' },
      { name: 'Weaviate', icon: '🕸️', description: 'Vector Search Engine', website: 'https://weaviate.io', color: '#00D2FF' },
      { name: 'Chroma', icon: '🎨', description: 'AI-native Vector Database', website: 'https://trychroma.com', color: '#FF6B6B' },
      { name: 'Ollama', icon: '🦙', description: 'Run LLMs Locally', website: 'https://ollama.ai', color: '#000000' },
      { name: 'Together AI', icon: '🤝', description: 'Decentralized AI Cloud', website: 'https://together.ai', color: '#6366F1' },
      { name: 'Perplexity', icon: '🔍', description: 'AI-powered Search', website: 'https://perplexity.ai', color: '#20B2AA' }
    ]
  },
  {
    title: 'MLOps & Production',
    description: (
      <>
        Production-ready ML systems with CI/CD pipelines, monitoring, scaling,
        and governance. From infrastructure setup to automated maintenance workflows.
      </>
    ),
    link: '/mlops-lifecycle',
    gradient: 'linear-gradient(135deg, #3182CE 0%, #2C5282 100%)',
    libraries: [
      { name: 'Kubeflow', icon: '☸️', description: 'ML Workflows on Kubernetes', website: 'https://kubeflow.org', color: '#326CE5' },
      { name: 'MLflow', icon: '🔄', description: 'ML Lifecycle Management', website: 'https://mlflow.org', color: '#0194E2' },
      { name: 'DVC', icon: '📊', description: 'Data Version Control', website: 'https://dvc.org', color: '#945ECF' },
      { name: 'Apache Airflow', icon: '🌪️', description: 'Workflow Orchestration', website: 'https://airflow.apache.org', color: '#017CEE' },
      { name: 'Prefect', icon: '🔄', description: 'Workflow Management', website: 'https://prefect.io', color: '#FF6B35' },
      { name: 'Weights & Biases', icon: '📊', description: 'Experiment Tracking', website: 'https://wandb.ai', color: '#FFBE00' },
      { name: 'Neptune', icon: '🔱', description: 'ML Metadata Store', website: 'https://neptune.ai', color: '#1BA1E2' },
      { name: 'Docker', icon: '🐳', description: 'Containerization', website: 'https://docker.com', color: '#0db7ed' },
      { name: 'Kubernetes', icon: '☸️', description: 'Container Orchestration', website: 'https://kubernetes.io', color: '#326CE5' },
      { name: 'Seldon', icon: '🎯', description: 'ML Deployment Platform', website: 'https://seldon.io', color: '#326CE5' },
      { name: 'BentoML', icon: '🍱', description: 'ML Model Serving', website: 'https://bentoml.org', color: '#FF6B35' },
      { name: 'Ray', icon: '☀️', description: 'Distributed AI Framework', website: 'https://ray.io', color: '#028CF0' },
      { name: 'Feast', icon: '🍽️', description: 'Feature Store', website: 'https://feast.dev', color: '#FF6B35' },
      { name: 'Great Expectations', icon: '✅', description: 'Data Quality', website: 'https://greatexpectations.io', color: '#FF6B6B' },
      { name: 'Evidently AI', icon: '📊', description: 'ML Monitoring', website: 'https://evidentlyai.com', color: '#FF4081' },
      { name: 'Prometheus', icon: '🔥', description: 'Monitoring & Alerting', website: 'https://prometheus.io', color: '#E6522C' },
      { name: 'Grafana', icon: '📊', description: 'Analytics & Monitoring', website: 'https://grafana.com', color: '#F46800' },
      { name: 'Terraform', icon: '🏗️', description: 'Infrastructure as Code', website: 'https://terraform.io', color: '#7B42BC' },
      { name: 'GitHub Actions', icon: '⚙️', description: 'CI/CD Platform', website: 'https://github.com/features/actions', color: '#2088FF' },
      { name: 'AWS SageMaker', icon: '☁️', description: 'ML Platform', website: 'https://aws.amazon.com/sagemaker/', color: '#FF9900' }
    ]
  }
];

function LifecycleSteps(): ReactNode {
  return (
    <section className={styles.stepsSection}>
      <div className="container">
        <div className={styles.stepsContent}>
          <Heading as="h2" className={styles.stepsTitle}>
            Accelerate AI development.
          </Heading>
          <p className={styles.stepsSubtitle}>
            Follow structured workflows with expert guidance for every step of your AI journey.
          </p>
          <div className={styles.stepsGrid}>
            <div className={styles.step}>
              <div className={styles.stepIcon}>01</div>
              <h3>Data Collection & Preparation</h3>
              <p>Gather, clean, and prepare quality datasets for your AI models.</p>
            </div>
            <div className={styles.step}>
              <div className={styles.stepIcon}>02</div>
              <h3>Feature Engineering</h3>
              <p>Transform raw data into meaningful features that improve model performance.</p>
            </div>
            <div className={styles.step}>
              <div className={styles.stepIcon}>03</div>
              <h3>Model Development</h3>
              <p>Build, train, and optimize AI models using industry best practices.</p>
            </div>
            <div className={styles.step}>
              <div className={styles.stepIcon}>04</div>
              <h3>Deployment & Monitoring</h3>
              <p>Deploy models to production and maintain them with continuous monitoring.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Feature({title, description, link, libraries, gradient}: FeatureItem) {
  const [showLibraries, setShowLibraries] = useState(false);
  const [hoveredLibrary, setHoveredLibrary] = useState<string | null>(null);

  return (
    <div className={clsx('col col--4')}>
      <div 
        className={styles.featureCard}
        onMouseEnter={() => setShowLibraries(true)}
        onMouseLeave={() => setShowLibraries(false)}
      >
        <Link to={link} className={styles.featureLink}>
          <div className={styles.featureIcon} style={{background: gradient}}>
            <div className={styles.iconPlaceholder}></div>
          </div>
          <div className={styles.featureContent}>
            <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
            <p className={styles.featureDescription}>{description}</p>
          </div>
          <div className={styles.featureArrow}>
            <span>→</span>
          </div>
        </Link>
        
        {showLibraries && (
          <div className={styles.librariesOverlay}>
            <div className={styles.librariesGrid}>
              {libraries.slice(0, 12).map((library, index) => (
                <div 
                  key={library.name}
                  className={styles.libraryItem}
                  style={{
                    '--delay': `${index * 0.1}s`,
                    '--color': library.color
                  } as React.CSSProperties}
                  onMouseEnter={() => setHoveredLibrary(library.name)}
                  onMouseLeave={() => setHoveredLibrary(null)}
                >
                  <div className={styles.libraryIcon}>
                    <span>{library.icon}</span>
                  </div>
                  <div className={styles.libraryName}>{library.name}</div>
                  {hoveredLibrary === library.name && (
                    <div className={styles.libraryTooltip}>
                      <div className={styles.tooltipContent}>
                        <strong>{library.name}</strong>
                        <p>{library.description}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className={styles.librariesCount}>
              +{libraries.length - 12} more libraries
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <>
      <section className={styles.features}>
        <div className="container">
          <div className={styles.featuresHeader}>
            <Heading as="h2" className={styles.featuresTitle}>
              The AI Development Stack
            </Heading>
            <p className={styles.featuresSubtitle}>
              Comprehensive documentation for every AI and machine learning domain
            </p>
          </div>
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </section>
      <LifecycleSteps />
    </>
  );
}
