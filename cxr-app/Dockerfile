# ---------- Stage 1: Build Angular app ----------
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies (include npm ci for clean install)
COPY package*.json .
RUN npm ci

# Copy source & build
COPY . .
RUN npm run build -- --output-path=dist/cxr-app/browser --configuration=production

# ---------- Stage 2: Serve with Nginx ----------
FROM nginx:stable-alpine

# Remove default static assets and set permissions
RUN rm -rf /usr/share/nginx/html/* && \
    chmod 755 /usr/share/nginx/html

# Copy Angular files from builder stage
COPY --from=builder --chown=nginx:nginx /app/dist/cxr-app/browser/* /usr/share/nginx/html/
# Overwrite default Nginx config
COPY ./default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
