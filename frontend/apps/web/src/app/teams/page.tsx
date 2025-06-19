"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { PageHeader } from "@/components/page-header";
import { Team } from "@nba/api/types";
import Link from "next/link";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";

export default function TeamsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["teams"],
    queryFn: () => apiClient.getTeams(30),
  });

  return (
    <main className="container mx-auto px-4 py-8">
      <PageHeader
        title="NBA Teams"
        description="Explore all 30 teams in the league."
      />

      <div className="mt-8">
        {isLoading && <p>Loading teams...</p>}
        {data && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.items.map((team: Team) => (
              <Link key={team.team_id} href={`/teams/${team.team_id}`}>
                <Card className="hover:bg-accent transition-colors">
                  <CardHeader>
                    <CardTitle>{team.full_name}</CardTitle>
                    <CardDescription>{team.team_abbrev}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}