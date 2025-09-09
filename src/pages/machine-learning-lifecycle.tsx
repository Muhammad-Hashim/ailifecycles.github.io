import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './machine-learning-lifecycle.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

const steps = [
  {
    title: 'Overview',
    description: 'Introduction to the Machine Learning Lifecycle and its importance in building robust ML systems.',
    link: '/docs/machine-learning-lifecycle/overview',
  },
  {
    title: 'Data Collection & Preparation',
    description: 'Strategies for gathering, cleaning, and preparing data for machine learning models.',
    link: '/docs/machine-learning-lifecycle/data-collection-preparation',
  },
  {
    title: 'Feature Engineering',
    description: 'Techniques for creating and selecting features that improve model performance.',
    link: '/docs/machine-learning-lifecycle/feature-engineering',
  },
  {
    title: 'Model Development',
    description: 'Designing and implementing machine learning models using appropriate algorithms.',
    link: '/docs/machine-learning-lifecycle/model-development',
  },
  {
    title: 'Training & Validation',
    description: 'Training models on data and validating their performance using various metrics.',
    link: '/docs/machine-learning-lifecycle/training-validation',
  },
  {
    title: 'Deployment',
    description: 'Strategies for deploying machine learning models into production environments.',
    link: '/docs/machine-learning-lifecycle/deployment',
  },
  {
    title: 'Monitoring & Maintenance',
    description: 'Continuous monitoring and maintenance of deployed ML models for optimal performance.',
    link: '/docs/machine-learning-lifecycle/monitoring-maintenance',
  },
  {
    title: 'Resources',
    description: 'Additional resources, tools, and best practices for machine learning development.',
    link: '/docs/machine-learning-lifecycle/resources',
  },
  {
    title: 'Roadmaps & Learning Paths',
    description: 'Structured learning paths and roadmaps for mastering machine learning concepts.',
    link: '/docs/machine-learning-lifecycle/roadmaps-learning-paths',
  },
];

export default function MachineLearningLifecycle() {
  return (
    <Layout
      title="Machine Learning Lifecycle"
      description="Complete guide to the Machine Learning Lifecycle from data collection to deployment and maintenance."
    >
      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <div className={styles.robotContainer}>
              <img 
                src={useBaseUrl('/img/robots/ml-robot.svg')} 
                alt="Machine Learning Robot"
                className={styles.robotImage}
              />
            </div>
            <h1>Machine Learning Lifecycle</h1>
            <p>
              A comprehensive guide through all stages of developing and deploying machine learning systems.
              Follow the structured approach to build reliable and scalable ML solutions.
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
