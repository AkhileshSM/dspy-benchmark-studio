FROM node:18-alpine AS build
WORKDIR /app

# Install dependencies and build
COPY package*.json ./
RUN npm install --silent --no-audit --no-fund
COPY . .
RUN npm run build

# Serve with nginx
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
