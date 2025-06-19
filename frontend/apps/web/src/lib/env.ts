import { z } from "zod";

const clientEnvSchema = z.object({
  NEXT_PUBLIC_API_URL: z.string().url(),
  NEXT_PUBLIC_SERVER_URL: z.string().url(),
});

const parsedEnv = clientEnvSchema.safeParse({
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  NEXT_PUBLIC_SERVER_URL: process.env.NEXT_PUBLIC_SERVER_URL,
});

if (!parsedEnv.success) {
  console.error(
    "Invalid client environment variables:",
    parsedEnv.error.flatten().fieldErrors,
  );
}

export const env = parsedEnv.success ? parsedEnv.data : {
    NEXT_PUBLIC_API_URL: "",
    NEXT_PUBLIC_SERVER_URL: "",
};