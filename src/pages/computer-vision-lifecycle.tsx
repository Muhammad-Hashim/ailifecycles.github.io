import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './computer-vision-lifecycle.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function ComputerVisionLifecycle() {
  return (
    <Layout
      title="Computer Vision Lifecycle"
      description="Complete guide to the Computer Vision Lifecycle from data collection to model deployment."
    >
      <main className={styles.main}>
        <div className="container">
          <div className={styles.header}>
            <div className={styles.robotContainer}>
              <img 
                src={useBaseUrl('/img/robots/cv-robot.svg')} 
                alt="Computer Vision Robot"
                className={styles.robotImage}
              />
            </div>
            <h1>Computer Vision Lifecycle</h1>
            <p>
              A comprehensive guide to developing and deploying computer vision systems.
              Learn about image processing, object detection, classification, and more.
            </p>
          </div>

          <div className={styles.content}>
            <div className={styles.overviewCard}>
              <h2>Overview</h2>
              <p>
                Get started with the fundamentals of computer vision lifecycle including
                data collection strategies, model architectures, training techniques,
                and deployment considerations for vision applications.
              </p>
              <Link to="/docs/computer-vision-lifecycle/overview" className={styles.link}>
                Read the Overview →
              </Link>
            </div>

            <div className={styles.comingSoon}>
              <h3>Coming Soon</h3>
              <p>
                Detailed guides for specific computer vision topics including:
              </p>
              <ul>
                <li>Image Data Collection & Preparation</li>
                <li>Model Architectures (CNNs, Transformers)</li>
                <li>Object Detection & Segmentation</li>
                <li>Training & Validation Strategies</li>
                <li>Model Optimization & Deployment</li>
                <li>Real-time Vision Applications</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
