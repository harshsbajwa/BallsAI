"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@nba/ui/table";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { PageHeader } from "@/components/page-header";
import { apiClient } from "@/lib/api-client";

interface TeamPageProps {
  params: {
    id: string;
  };
}

export default function TeamPage({ params }: TeamPageProps) {
  const teamId = parseInt(params.id, 10);

  const { data: teamRoster, isLoading } = useQuery({
    queryKey: ["team", teamId, "roster"],
    queryFn: () => apiClient.getTeamRoster(teamId),
    enabled: !!teamId,
  });

  if (isLoading) return <p>Loading team details...</p>;
  if (!teamRoster) return <p>Team not found.</p>;

  return (
    <div className="container mx-auto px-4 py-8">
      <PageHeader
        title={teamRoster.team.full_name}
        description={`Current Roster for the ${teamRoster.team.team_name}`}
      />

      <div className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Team Roster</CardTitle>
            <CardDescription>
              Click a player to view their details.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Player</TableHead>
                  <TableHead>Position</TableHead>
                  <TableHead>Height</TableHead>
                  <TableHead>Weight</TableHead>
                  <TableHead>Draft Year</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {teamRoster.roster.map((player) => (
                  <TableRow key={player.person_id}>
                    <TableCell className="font-medium">
                      <Link
                        href={`/players/${player.person_id}`}
                        className="hover:underline"
                      >
                        {player.full_name}
                      </Link>
                    </TableCell>
                    <TableCell>{player.primary_position}</TableCell>
                    <TableCell>{player.height}`&quot;`</TableCell>
                    <TableCell>{player.body_weight} lbs</TableCell>
                    <TableCell>{player.draft_year || "Undrafted"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}