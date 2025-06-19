import ApiClient from "@nba/api/client";
import { env } from "./env";

export const apiClient = new ApiClient(env.NEXT_PUBLIC_API_URL);