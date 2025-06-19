import { db } from "@nba/db/client";
import { users } from "@nba/db/schema";
import bcrypt from "bcryptjs";
import { z } from "zod";
import { NextResponse } from "next/server";

const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { email, password } = signupSchema.parse(body);

    const existingUser = await db.query.users.findFirst({
      where: (u, { eq }) => eq(u.email, email),
    });

    if (existingUser) {
      return NextResponse.json(
        { message: "User with this email already exists" },
        { status: 409 },
      );
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    await db.insert(users).values({ email, hashedPassword });

    return NextResponse.json(
      { message: "User created successfully" },
      { status: 201 },
    );
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ errors: error.errors }, { status: 400 });
    }
    return NextResponse.json(
      { message: "An internal server error occurred" },
      { status: 500 },
    );
  }
}