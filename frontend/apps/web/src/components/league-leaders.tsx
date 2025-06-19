"use client";

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
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
import Link from "next/link";

export function LeagueLeaders() {
  const currentSeasonYear = new Date().getFullYear();
  const { data, isLoading } = useQuery({
    queryKey: ["analytics", "league-leaders", currentSeasonYear],
    queryFn: () => apiClient.getLeagueLeaders(currentSeasonYear - 1, 5),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scoring Leaders</CardTitle>
        <CardDescription>
          Top 5 points per game leaders for the {currentSeasonYear - 1}-
          {currentSeasonYear} season.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <p>Loading leaders...</p>}
        {data && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Rank</TableHead>
                <TableHead>Player</TableHead>
                <TableHead>Team</TableHead>
                <TableHead className="text-right">PPG</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.map((leader, index) => (
                <TableRow key={leader.player.person_id}>
                  <TableCell>{index + 1}</TableCell>
                  <TableCell className="font-medium">
                    <Link
                      href={`/players/${leader.player.person_id}`}
                      className="hover:underline"
                    >
                      {leader.player.full_name}
                    </Link>
                  </TableCell>
                  <TableCell>{leader.team.team_abbrev}</TableCell>
                  <TableCell className="text-right font-bold">
                    {leader.avg_points.toFixed(1)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}