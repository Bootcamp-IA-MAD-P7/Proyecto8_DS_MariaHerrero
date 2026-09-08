FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci

COPY . .

ARG VITE_API_URL=/api/v1
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

FROM nginx:alpine

ENV PORT=80 \
    BACKEND_HOST=backend \
    BACKEND_PORT=8000 \
    NGINX_ENVSUBST_FILTER="^(PORT|BACKEND_HOST|BACKEND_PORT)$"

COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
