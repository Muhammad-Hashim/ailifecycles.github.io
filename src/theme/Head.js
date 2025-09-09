import React from 'react';
import {useColorMode} from '@docusaurus/theme-common';

export default function Head() {
  const {colorMode} = useColorMode();
  
  return (
    <>
      <link 
        rel="stylesheet" 
        href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap" 
      />
    </>
  );
}
