import ApiClient from "@nba/api/client";

export const apiClient = new ApiClient(process.env.EXPO_PUBLIC_API_URL!);