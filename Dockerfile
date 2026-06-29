FROM nginx:alpine

# Copy the static frontend files to nginx's default public directory
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY quality_report.json /usr/share/nginx/html/

# Expose port 80 for Railway's routing
EXPOSE 80
