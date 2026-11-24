'use strict';
(() => {let saved;try{saved=localStorage.getItem('countlab-theme')}catch{}const theme=['light','dark'].includes(saved)?saved:'dark';document.documentElement.dataset.theme=theme;document.documentElement.style.colorScheme=theme;})();
