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

import { PageHeader } from "@/components/page-header";
import { apiClient } from "@/lib/api-client";

interface PlayerPageProps {
  params: {
    id: string;
  };
}

export default function PlayerPage({ params }: PlayerPageProps) {
  const playerId = parseInt(params.id, 10);

  const { data: player, isLoading: playerLoading } = useQuery({
    queryKey: ["player", playerId],
    queryFn: () => apiClient.getPlayer(playerId),
    enabled: !!playerId,
  });

  const { data: playerStats, isLoading: statsLoading } = useQuery({
    queryKey: ["player", playerId, "stats"],
    queryFn: () => apiClient.getPlayerStats(playerId, 10),
    enabled: !!playerId,
  });

  if (playerLoading) return <p>Loading player details...</p>;
  if (!player) return <p>Player not found.</p>;

  return (
    <div className="container mx-auto px-4 py-8">
      <PageHeader
        title={player.full_name}
        description={`${player.primary_position} ${
          player.draft_year ? `• Draft ${player.draft_year}` : ""
        }`}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-8">
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Player Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Height</span>
                <span>{player.height}`&quot;`</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Weight</span>
                <span>{player.body_weight} lbs</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Born</span>
                <span>
                  {player.birthdate
                    ? new Date(player.birthdate).toLocaleDateString()
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Country</span>
                <span>{player.country}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Draft</span>
                <span>
                  {player.draft_year
                    ? `${player.draft_year} R${player.draft_round} P${player.draft_number}`
                    : "Undrafted"}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="md:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Recent Game Stats</CardTitle>
              <CardDescription>Last 10 games played.</CardDescription>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <p>Loading stats...</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead className="text-right">PTS</TableHead>
                      <TableHead className="text-right">REB</TableHead>
                      <TableHead className="text-right">AST</TableHead>
                      <TableHead className="text-right">STL</TableHead>
                      <TableHead className="text-right">BLK</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {playerStats?.items.map((stat) => (
                      <TableRow key={stat.game_id}>
                        <TableCell>
                          {new Date(stat.game_date).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right font-medium">
                          {stat.points}
                        </TableCell>
                        <TableCell className="text-right">
                          {stat.rebounds_total}
                        </TableCell>
                        <TableCell className="text-right">
                          {stat.assists}
                        </TableCell>
                        <TableCell className="text-right">
                          {stat.steals}
                        </TableCell>
                        <TableCell className="text-right">
                          {stat.blocks}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}