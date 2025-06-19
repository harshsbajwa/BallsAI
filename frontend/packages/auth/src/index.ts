import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { db } from "@nba/db/client";
import { users } from "@nba/db/schema";
import { eq } from "drizzle-orm";
import { z } from "zod";
import bcrypt from "bcryptjs";

export const auth: ReturnType<typeof betterAuth> = betterAuth({
  database: drizzleAdapter(db, {
    provider: "pg",
  }),
  secret: process.env.AUTH_SECRET,
  pages: {
    signIn: "/login",
  },
  callbacks: {
    authorize: async (credentials: Record<string, unknown>) => {
      const { email, password } = z
        .object({
          email: z.string().email(),
          password: z.string(),
        })
        .parse(credentials);

      const user = await db.query.users.findFirst({
        // Explicitly use the imported 'eq' and 'users' schema
        where: eq(users.email, email),
      });

      if (!user || !user.hashedPassword) {
        return null;
      }

      const isValidPassword = await bcrypt.compare(
        password,
        user.hashedPassword,
      );

      if (!isValidPassword) {
        return null;
      }

      return { id: user.id, email: user.email, name: user.name };
    },
  },
});