import { auth } from "@nba/auth";
import { Button } from "@nba/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import { redirect } from "next/navigation";
import { headers } from "next/headers";

export default async function ProfilePage() {
  const session = await auth.api.getSession({
    headers: headers(),
  });

  if (!session?.user) {
    redirect("/login");
  }

  return (
    <div className="container mx-auto flex h-[calc(100vh-10rem)] items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Profile</CardTitle>
          <CardDescription>
            Welcome back, you are logged in.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm text-muted-foreground">Email</p>
            <p>{session.user.email}</p>
          </div>
          <form
            action={async () => {
              "use server";
              await auth.api.signOut({
                headers: headers()
              });
              redirect("/");
            }}
          >
            <Button type="submit" variant="destructive" className="w-full">
              Sign Out
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}