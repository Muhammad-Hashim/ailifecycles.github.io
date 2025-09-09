import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './mlops-lifecycle.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function MlopsLifecycle() {
  return (
    <Layout
      title="MLOps Lifecycle"
      description="Complete guide to the MLOps Lifecycle for production ML systems."
    >
      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <div className={styles.robotContainer}>
              <img 
                src={useBaseUrl('/img/robots/mlops-robot.svg')} 
                alt="MLOps Robot"
                className={styles.robotImage}
              />
            </div>
            <h1>MLOps Lifecycle</h1>
            <p>
              A comprehensive guide to operationalizing machine learning systems.
              Learn about CI/CD, monitoring, scaling, and governance.
            </p>
          </div>

          <div className={styles.content}>
            <div className={styles.overviewCard}>
              <h2>Overview</h2>
              <p>
                Get started with the fundamentals of MLOps lifecycle including
                infrastructure setup, automated pipelines, model monitoring,
                and production deployment strategies.
              </p>
              <Link to="/docs/mlops-lifecycle/overview" className={styles.link}>
                Read the Overview →
              </Link>
            </div>

            <div className={styles.comingSoon}>
              <h3>Coming Soon</h3>
              <p>
                Detailed guides for MLOps topics including:
              </p>
              <ul>
                <li>CI/CD Pipelines for ML</li>
                <li>Model Monitoring & Alerting</li>
                <li>Infrastructure & Scaling</li>
                <li>Data Pipeline Automation</li>
                <li>Model Governance & Compliance</li>
                <li>Production Deployment Strategies</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
