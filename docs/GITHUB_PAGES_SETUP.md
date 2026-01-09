# 🌐 Guía de Configuración de GitHub Pages

Esta guía te ayudará a configurar y publicar tu página web en GitHub Pages.

## 📋 Paso 1: Configurar la Información de GitHub

1. Abre el archivo `docs/index.html` en tu editor
2. Busca la sección de configuración al inicio del archivo (líneas 257-261)
3. Reemplaza los valores de configuración:

```javascript
const GITHUB_CONFIG = {
    username: 'TU_USUARIO_GITHUB',  // ← Cambia esto por tu usuario de GitHub
    repository: 'Clasificaci-n-Sem-ntica-',  // ← Cambia esto por el nombre de tu repositorio
    repositoryFullName: 'TU_USUARIO_GITHUB/Clasificaci-n-Sem-ntica-'  // ← Usuario/Repositorio
};
```

**Ejemplo:**
```javascript
const GITHUB_CONFIG = {
    username: 'juanperez',
    repository: 'Clasificaci-n-Sem-ntica-',
    repositoryFullName: 'juanperez/Clasificaci-n-Sem-ntica-'
};
```

## 🚀 Paso 2: Activar GitHub Pages en tu Repositorio

### Opción A: Usando la Interfaz Web de GitHub (Recomendado)

1. Ve a tu repositorio en GitHub
2. Haz clic en **Settings** (Configuración) en la barra superior del repositorio
3. En el menú lateral izquierdo, busca y haz clic en **Pages**
4. En la sección **Source** (Fuente):
   - Selecciona **Deploy from a branch** (Desplegar desde una rama)
   - Selecciona la rama: **main** (o **master** si es tu rama principal)
   - Selecciona la carpeta: **/docs**
   - Haz clic en **Save** (Guardar)
5. Espera unos minutos mientras GitHub genera tu sitio
6. Tu página estará disponible en: `https://TU_USUARIO_GITHUB.github.io/NOMBRE_REPOSITORIO/`

### Opción B: Usando GitHub Actions (Automático)

Si prefieres usar el workflow automático que está configurado:

1. Los archivos ya están listos en `.github/workflows/deploy-pages.yml`
2. Solo necesitas activar GitHub Pages desde Settings → Pages
3. Selecciona **GitHub Actions** como fuente (en lugar de "Deploy from a branch")
4. El workflow se ejecutará automáticamente cada vez que hagas push a la rama main

## ✅ Paso 3: Verificar que Todo Funciona

1. Después de activar GitHub Pages, espera 2-5 minutos
2. Ve a la URL que GitHub te proporcionó (generalmente aparece en Settings → Pages)
3. Deberías ver tu página con toda la información del proyecto

## 🔧 Paso 4: Actualizar los Enlaces

Una vez que actualices la configuración en `docs/index.html`, todos los enlaces se actualizarán automáticamente:

- ✅ Enlace al repositorio
- ✅ Enlace de descarga del código
- ✅ Enlace al README
- ✅ Enlace a Issues
- ✅ Comandos de clonación

## 🐛 Solución de Problemas

### La página no se actualiza
- Espera unos minutos (puede tardar hasta 10 minutos)
- Verifica que el archivo `docs/index.html` esté en la rama correcta
- Asegúrate de que GitHub Pages esté activado en Settings → Pages

### Los enlaces no funcionan
- Verifica que hayas actualizado la configuración en `docs/index.html`
- Asegúrate de que el formato del nombre de usuario y repositorio sea correcto
- Los nombres son sensibles a mayúsculas/minúsculas

### Error 404
- Verifica que el archivo `docs/index.html` exista
- Asegúrate de que el archivo `docs/.nojekyll` exista (previene problemas con Jekyll)
- Verifica que la carpeta configurada en GitHub Pages sea `/docs`

### El workflow de GitHub Actions falla
- Ve a la pestaña **Actions** en tu repositorio
- Revisa los logs del workflow fallido
- Asegúrate de que tengas permisos para escribir en Pages (Settings → Actions → General → Workflow permissions)

## 📝 Archivos Importantes

- `docs/index.html` - Página principal de GitHub Pages
- `docs/.nojekyll` - Evita que GitHub procese el sitio con Jekyll
- `.github/workflows/deploy-pages.yml` - Workflow automático para despliegue (opcional)

## 🎉 ¡Listo!

Una vez configurado, tu página estará disponible públicamente en GitHub Pages y se actualizará automáticamente cada vez que hagas cambios en la carpeta `docs/`.
