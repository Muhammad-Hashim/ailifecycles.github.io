import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './nlp-lifecycle.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

const steps = [
  {
    title: 'Overview',
    description: 'Introduction to the Natural Language Processing Lifecycle and its applications.',
    link: '/docs/nlp-lifecycle/overview',
  },
  {
    title: 'Problem Definition',
    description: 'Defining NLP problems, understanding requirements, and setting project goals.',
    link: '/docs/nlp-lifecycle/problem-definition',
  },
  {
    title: 'Data Collection',
    description: 'Strategies for collecting and sourcing text data for NLP tasks.',
    link: '/docs/nlp-lifecycle/data-collection',
  },
  {
    title: 'Text Exploration & Analysis',
    description: 'Analyzing text data to understand patterns, distributions, and quality.',
    link: '/docs/nlp-lifecycle/text-exploration-analysis',
  },
  {
    title: 'Text Preprocessing & Preparation',
    description: 'Cleaning, normalizing, and preparing text data for model training.',
    link: '/docs/nlp-lifecycle/text-preprocessing-preparation',
  },
  {
    title: 'Feature Engineering',
    description: 'Creating meaningful features from text data for machine learning models.',
    link: '/docs/nlp-lifecycle/feature-engineering',
  },
  {
    title: 'Model Training',
    description: 'Training NLP models using various algorithms and architectures.',
    link: '/docs/nlp-lifecycle/model-training',
  },
  {
    title: 'Model Evaluation',
    description: 'Evaluating model performance using appropriate metrics and techniques.',
    link: '/docs/nlp-lifecycle/model-evaluation',
  },
  {
    title: 'Model Deployment',
    description: 'Deploying NLP models into production environments with proper monitoring.',
    link: '/docs/nlp-lifecycle/model-deployment',
  },
];

export default function NlpLifecycle() {
  return (
    <Layout
      title="Natural Language Processing Lifecycle"
      description="Complete guide to the NLP Lifecycle from problem definition to model deployment."
    >
      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <div className={styles.robotContainer}>
              <img 
                src={useBaseUrl('/img/robots/nlp-robot.svg')} 
                alt="NLP Robot"
                className={styles.robotImage}
              />
            </div>
            <h1>Natural Language Processing Lifecycle</h1>
            <p>
              A comprehensive guide through all stages of developing and deploying natural language processing systems.
              Follow the structured approach to build reliable NLP solutions.
            </p>
          </div>

          <div className={styles.stepsGrid}>
            {steps.map((step, index) => (
              <div key={index} className={styles.stepCard}>
                <div className={styles.stepNumber}>{index + 1}</div>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
                <Link to={step.link} className={styles.stepLink}>
                  Learn More →
                </Link>
              </div>
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}
