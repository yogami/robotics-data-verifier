FROM nginx:alpine

# Copy the static frontend files
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY quality_report.json /usr/share/nginx/html/

# Create a template so Nginx dynamically listens on the port Railway provides ($PORT)
RUN mkdir -p /etc/nginx/templates && \
    echo 'server { listen ${PORT}; location / { root /usr/share/nginx/html; index index.html; } }' > /etc/nginx/templates/default.conf.template
